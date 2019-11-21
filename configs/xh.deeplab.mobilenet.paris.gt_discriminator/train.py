import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
# from mypath import Path
from common import config
from data import make_data_loader
from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.discriminator import FCDiscriminator
import json
import visdom
import torch
from torch.autograd import Variable
from utils.argmax import ArgMax


class Trainer(object):
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.vis = visdom.Visdom(env=os.getcwd().split('/')[-1])
        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(config)
        self.gpu = args.gpu

        # labels for adversarial training
        self.gt_label = 0
        self.prediction_label = 1
        self.argmax = ArgMax()

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=config.backbone,
                        output_stride=config.out_stride,
                        sync_bn=config.sync_bn,
                        freeze_bn=config.freeze_bn)
        model_D = FCDiscriminator(num_classes=self.nclass)
        model_D.train()

        train_params = [{'params': model.get_1x_lr_params(), 'lr': config.lr},
                        {'params': model.get_10x_lr_params(), 'lr': config.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
        optimizer_D.zero_grad()

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=config.loss)
        self.criterion_D = torch.nn.BCEWithLogitsLoss()
        self.model, self.optimizer = model, optimizer
        self.model_D, self.optimizer_D = model_D, optimizer_D

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(config.lr_scheduler, config.lr,
                                      config.epochs, len(self.train_loader),
                                      config.lr_step, config.warmup_epochs)

        # Using cuda
        if args.cuda:
            # self.model = torch.nn.DataParallel(self.model)
            # patch_replication_callback(self.model)
            # self.model_D = torch.nn.DataParallel(self.model)
            # patch_replication_callback(self.model_D)
            # cudnn.benchmark = True
            self.model = self.model.cuda()
            self.model_D = self.model_D.cuda()
            self.argmax = self.argmax

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint, map_location=torch.device('cpu'))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))

    def training(self, epoch):
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        seg_loss = 0.0
        adv_loss = 0.0
        D_loss = 0.0

        self.model.train()

        for i, sample in enumerate(tbar):
            iter = epoch * len(self.train_loader) + i
            self.vis.line(X=torch.tensor([iter]), Y=torch.tensor([self.optimizer.param_groups[0]['lr']]),
                          win='lr_seg', opts=dict(title='lr', xlabel='iter', ylabel='lr'),
                          update='append' if iter > 0 else None)
            self.vis.line(X=torch.tensor([iter]), Y=torch.tensor([self.optimizer_D.param_groups[0]['lr']]),
                          win='lr_adv', opts=dict(title='lr', xlabel='iter', ylabel='lr'),
                          update='append' if iter > 0 else None)

            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            # train the segmentation network
            # don't accumulate grads in D
            for param in self.model_D.parameters():
                param.requires_grad = False

            # seg loss
            output = self.model(image)
            loss1 = self.criterion(output, target)
            loss1.backward()
            self.optimizer.step()
            seg_loss += loss1.item()

            # adv loss
            print(self.argmax.apply(output))
            D_out = self.model_D(self.argmax.apply(output))
            loss2 = self.criterion_D(D_out,
                                     Variable(torch.FloatTensor(D_out.data.size()).fill_(self.gt_label)).cuda(self.gpu))
            loss2.backward()
            self.optimizer_D.step()
            adv_loss += loss2.item()

            # train the discriminator
            # bring back requires_grad
            for param in self.model_D.parameters():
                param.requires_grad = True

            # train_with_prediction
            output = output.detach()
            D_out1 = self.model_D(self.argmax.apply(output))

            loss_D1 = self.criterion_D(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(self.prediction_label)).cuda(
                                           self.gpu))

            loss_D1.backward()
            D_loss += loss_D1.data.cpu().numpy()

            # train with gt
            D_out2 = self.model_D(target)
            loss_D2 = self.criterion_D(D_out2,
                                       Variable(torch.FloatTensor(D_out2.data.size()).fill_(self.gt_label)).cuda(
                                           self.gpu))

            loss_D2.backward()
            D_loss += loss_D2.data.cpu().numpy()

            tbar.set_description('[Train] Seg loss: %.3f, Adv loss: %.3f, D loss: %.3f' \
                                 % (seg_loss / (i + 1), adv_loss / (i + 1), D_loss / (i + 1)))

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config.batch_size + image.data.shape[0]))
        print('Seg loss: %.3f, Adv loss: %.3f, D loss: %.3f' \
              % (seg_loss / (i + 1), adv_loss / (i + 1), D_loss / (i + 1)))

        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([seg_loss]), win='seg_loss', name='train',
                      opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                      update='append' if epoch > 0 else None)
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([adv_loss]), win='adv_loss', name='train',
                      opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                      update='append' if epoch > 0 else None)
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([D_loss]), win='D_loss', name='train',
                      opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                      update='append' if epoch > 0 else None)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            seg_loss = self.criterion(output, target)
            seg_loss += seg_loss.item()
            output = output.detach()
            D_out = self.model_D(self.argmax(output))
            adv_loss = self.criterion_D(D_out,
                                        Variable(torch.FloatTensor(D_out.data.size()).fill_(self.gt_label)).cuda(
                                            self.gpu))
            tbar.set_description('[Test] Seg loss: %.3f, Adv loss: %.3f' % (seg_loss / (i + 1), adv_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Building_Acc()
        # Acc_class = self.evaluator.Pixel_Accuracy_Class()
        IoU = self.evaluator.Building_IoU()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config.batch_size + image.data.shape[0]))
        print("Acc:{}, IoU:{}, mIoU:{}".format(Acc, IoU, mIoU))
        print('Seg loss: %.3f, Adv loss: %.3f' % (seg_loss / (i + 1), adv_loss / (i + 1)))

        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([seg_loss]), win='seg_loss', name='val',
                      update='append')
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([adv_loss]), win='adv_loss', name='val',
                      update='append')
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([Acc]), win='metrics', name='acc',
                      opts=dict(title='metrics', xlabel='epoch', ylabel='performance'),
                      update='append' if epoch > 0 else None)
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([IoU]), win='metrics', name='IoU',
                      update='append')
        self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([mIoU]), win='metrics', name='mIoU',
                      update='append')

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            print('Saving state, epoch:', epoch)
            torch.save(self.model.state_dict(), self.args.save_folder + 'models/'
                       + 'epoch' + str(epoch) + '.pth')
            loss_file = {'Acc': Acc, 'IoU': IoU, 'mIoU': mIoU}
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
    parser.add_argument('--resume', type=str,
                        default=None,
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
    if args.cuda:
        print('Using cuda device:', args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
