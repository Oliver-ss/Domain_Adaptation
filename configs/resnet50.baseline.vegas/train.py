import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from common import config
from data import make_data_loader, make_target_data_loader
from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from tensorboardX import SummaryWriter

import json
import visdom
import torch

class Trainer(object):
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.visdom = args.visdom
        if args.visdom:
            self.vis = visdom.Visdom(env=os.getcwd().split('/')[-1],port=8888)
        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(config)

        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                        backbone=config.backbone,
                        output_stride=config.out_stride,
                        sync_bn=config.sync_bn,
                        freeze_bn=config.freeze_bn)


        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': config.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': config.lr * 10}]

        # Define Optimizer
        self.optimizer = torch.optim.SGD(train_params, momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=config.loss)
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(config.lr_scheduler, config.lr,
                                      config.epochs, len(self.train_loader),
                                      config.lr_step, config.warmup_epochs)
        self.writer = SummaryWriter('train_log')
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            # cudnn.benchmark = True
            self.model = self.model.cuda()


        self.best_pred_source = 0.0
        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint, map_location=torch.device('cpu'))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            itr = epoch * len(self.train_loader) + i
            if self.visdom:
                self.vis.line(X=torch.tensor([itr]), Y=torch.tensor([self.optimizer.param_groups[0]['lr']]),
                          win='lr', opts=dict(title='lr', xlabel='iter', ylabel='lr'),
                          update='append' if itr>0 else None)
            A_image, A_target = sample['image'], sample['label']

            if self.args.cuda:
                A_image, A_target = A_image.cuda(), A_target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred_source, 0)

            A_output, A_feat, A_low_feat = self.model(A_image)

            self.optimizer.zero_grad()


            # Supervised loss
            seg_loss = self.criterion(A_output, A_target)
            loss = seg_loss
            loss.backward()


            self.optimizer.step()

            train_loss += seg_loss.item()
            self.writer.add_scalar('Train/Loss', loss.item(), itr)
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config.batch_size + A_image.data.shape[0]))
        print('Seg Loss: %.3f' % seg_loss_sum)

        if self.visdom:
            self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([seg_loss_sum]), win='train_loss', name='Seg_loss',
                          opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                          update='append' if epoch > 0 else None)

    def validation(self, epoch):
        def get_metrics(tbar, if_source=False):
            self.evaluator.reset()
            test_loss = 0.0
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']

                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                with torch.no_grad():
                    output, low_feat, feat = self.model(image)

                loss = self.criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()

                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)

                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)

            self.writer.add_scalar('Val/Loss', test_loss/(i+1), epoch)
            # Fast test during the training
            Acc = self.evaluator.Building_Acc()
            IoU = self.evaluator.Building_IoU()
            mIoU = self.evaluator.Mean_Intersection_over_Union()

            if if_source:
                print('Validation on source:')
            else:
                print('Validation on target:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config.batch_size + image.data.shape[0]))
            print("Acc:{}, IoU:{}, mIoU:{}".format(Acc, IoU, mIoU))
            print('Loss: %.3f' % test_loss)

            if if_source:
                names = ['source', 'source_acc', 'source_IoU', 'source_mIoU']
                self.writer.add_scalar('Val/SourceAcc', Acc, epoch)
                self.writer.add_scalar('Val/SourceIoU', IoU, epoch)
            else:
                names = ['target', 'target_acc', 'target_IoU', 'target_mIoU']
                self.writer.add_scalar('Val/TargetAcc', Acc, epoch)
                self.writer.add_scalar('Val/TargetIoU', IoU, epoch)

            # Draw Visdom
            if self.visdom:
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([test_loss]), win='val_loss', name=names[0],
                              update='append')
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([Acc]), win='metrics', name=names[1],
                              opts=dict(title='metrics', xlabel='epoch', ylabel='performance'),
                              update='append' if epoch > 0 else None)
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([IoU]), win='metrics', name=names[2],
                              update='append')
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([mIoU]), win='metrics', name=names[3],
                              update='append')

            return Acc, IoU, mIoU

        self.model.eval()
        tbar_source = tqdm(self.val_loader, desc='\r')
        tbar_target = tqdm(self.target_val_loader, desc='\r')
        s_acc, s_iou, s_miou = get_metrics(tbar_source, True)

        new_pred_source = s_iou

        if new_pred_source > self.best_pred_source:
            is_best = True
            self.best_pred_source = max(new_pred_source, self.best_pred_source)
            print('Saving state, epoch:', epoch)
            torch.save(self.model.module.state_dict(), self.args.save_folder + 'models/'
                       + 'epoch' + str(epoch) + '.pth')
        loss_file = {'s_Acc': s_acc, 's_IoU': s_iou, 's_mIoU': s_miou}
        with open(os.path.join(self.args.save_folder, 'eval', 'epoch' + str(epoch) + '.json'), 'w') as f:
            json.dump(loss_file, f)


def main():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    # training hyper params
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--save_folder', default='train_log/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--visdom', default=True, type=str2bool,
                        help='whether to Visdom')
    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    if not os.path.exists(args.save_folder + 'eval/'):
        os.mkdir(args.save_folder + 'eval/')
    # if not os.path.exists(args.save_folder + 'models/'):
    #     os.mkdir(args.save_folder + 'models/')
    if not os.path.exists('/usr/xtmp/satellite/train_models/' + os.getcwd().split('/')[-1]):
        os.mkdir('/usr/xtmp/satellite/train_models/' + os.getcwd().split('/')[-1])
        os.symlink('/usr/xtmp/satellite/train_models/' + os.getcwd().split('/')[-1], args.save_folder + 'models')
        print('Create soft link!')

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #if args.cuda:
    #    print('Using cuda device:', args.gpu)
    #    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(config, args)

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.config.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.config.epochs):
        trainer.training(epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        trainer.validation(epoch)


if __name__ == "__main__":
    main()
