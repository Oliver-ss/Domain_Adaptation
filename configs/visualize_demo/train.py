import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
# from mypath import Path
from common import config
from data import make_data_loader, make_target_data_loader
from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *
from utils.loss import SegmentationLosses, MinimizeEntropyLoss, BottleneckLoss, BottleneckClassLoss
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
import json
import visdom
import torch


class Trainer(object):
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.vis = visdom.Visdom(env=os.getcwd().split('/')[-1],port=8888)
        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(config)
        self.target_train_loader, self.target_val_loader, self.target_test_loader, _ = make_target_data_loader(config)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=config.backbone,
                        output_stride=config.out_stride,
                        sync_bn=config.sync_bn,
                        freeze_bn=config.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': config.lr},
                        {'params': model.get_10x_lr_params(), 'lr': config.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=config.loss)
        self.model, self.optimizer = model, optimizer
        self.entropy_mini_loss = MinimizeEntropyLoss()
        #self.batchloss = BatchLoss()
        #self.bottleneck_loss = BottleneckLoss()
        self.bottleneck_loss = BottleneckClassLoss()

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(config.lr_scheduler, config.lr,
                                      config.epochs, len(self.train_loader),
                                      config.lr_step, config.warmup_epochs)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            # cudnn.benchmark = True
            self.model = self.model.cuda()

        self.best_pred_source = 0.0
        self.best_pred_target = 0.0
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
        train_loss, seg_loss_sum, bn_loss_sum, entropy_loss_sum = 0.0, 0.0, 0.0, 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        target_train_iterator = iter(self.target_train_loader)
        for i, sample in enumerate(tbar):
            itr = epoch * len(self.train_loader) + i
            self.vis.line(X=torch.tensor([itr]), Y=torch.tensor([self.optimizer.param_groups[0]['lr']]),
                          win='lr', opts=dict(title='lr', xlabel='iter', ylabel='lr'),
                          update='append' if itr>0 else None)
            A_image, A_target = sample['image'], sample['label']

            # Get one batch from target domain
            try:
                target_sample = next(target_train_iterator)
            except StopIteration:
                target_train_iterator = iter(self.target_train_loader)
                target_sample = next(target_train_iterator)

            B_image, B_target = target_sample['image'], target_sample['label']

            if self.args.cuda:
                A_image, A_target = A_image.cuda(), A_target.cuda()
                B_image, B_target = B_image.cuda(), B_target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred_source, self.best_pred_target)
            # Supervised loss
            self.optimizer.zero_grad()
            A_output, A_feat, A_low_feat = self.model(A_image)
            seg_loss = self.criterion(A_output, A_target)

            # Unsupervised bn loss
            B_output, B_feat, B_low_feat = self.model(B_image)
            B_label = torch.argmax(B_output, dim=1).float()
            bottleneck_loss = self.bottleneck_loss.loss(A_feat, B_feat, A_target, B_label) + self.bottleneck_loss.loss(A_low_feat, B_low_feat, A_target, B_label)

            # Unsupervised entropy minimization loss
            entropy_mini_loss = self.entropy_mini_loss.loss(B_output)
            loss = seg_loss + bottleneck_loss

            loss.backward()
            self.optimizer.step()

            seg_loss_sum += seg_loss.item()
            bn_loss_sum += bottleneck_loss.item()
            entropy_loss_sum += entropy_mini_loss.item()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config.batch_size + A_image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([seg_loss_sum]), win='train_loss', name='Seg_loss',
                      opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                      update='append' if epoch > 0 else None)
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([bn_loss_sum]), win='train_loss', name='BN_loss',
                      opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                      update='append' if epoch > 0 else None)
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([entropy_loss_sum]), win='train_loss', name='Entropy_loss',
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
                    output,_,_ = self.model(image)

                loss = self.criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)

                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)

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

            # Draw Visdom
            if if_source:
                names = ['source', 'source_acc', 'source_IoU', 'source_mIoU']
            else:
                names = ['target', 'target_acc', 'target_IoU', 'target_mIoU']

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
        #self.evaluator.reset()
        tbar_source = tqdm(self.val_loader, desc='\r')
        tbar_target = tqdm(self.target_val_loader, desc='\r')
        s_acc, s_iou, s_miou = get_metrics(tbar_source, True)
        t_acc, t_iou, t_miou = get_metrics(tbar_target, False)

        new_pred_source = s_iou
        new_pred_target = t_iou
        if new_pred_source > self.best_pred_source or new_pred_target > self.best_pred_target:
            is_best = True
            self.best_pred_source = max(new_pred_source, self.best_pred_source)
            self.best_pred_target = max(new_pred_target, self.best_pred_target)
            print('Saving state, epoch:', epoch)
            torch.save(self.model.module.state_dict(), self.args.save_folder + 'models/'
                       + 'epoch' + str(epoch) + '.pth')
            loss_file = {'s_Acc': s_acc, 's_IoU': s_iou, 's_mIoU': s_miou, 't_Acc':t_acc, 't_IoU':t_iou, 't_mIoU':t_miou}
            with open(os.path.join(self.args.save_folder, 'eval', 'epoch' + str(epoch) + '.json'), 'w') as f:
                json.dump(loss_file, f)


def main():
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
