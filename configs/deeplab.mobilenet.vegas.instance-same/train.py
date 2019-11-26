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
from model.discriminator import Discriminator
from utils.loss import SegmentationLosses, MinimizeEntropyLoss, BottleneckLoss, InstanceLoss
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.func import bce_loss, prob_2_entropy, flip
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
        self.target_train_loader, self.target_val_loader, self.target_test_loader, _ = make_target_data_loader(config)

        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                        backbone=config.backbone,
                        output_stride=config.out_stride,
                        sync_bn=config.sync_bn,
                        freeze_bn=config.freeze_bn)

        #self.D = Discriminator(num_classes=self.nclass, ndf=16)

        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': config.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': config.lr * 10}]

        # Define Optimizer
        self.optimizer = torch.optim.SGD(train_params, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        #self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.lr, betas=(0.9, 0.99))

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=config.loss)
        #self.model, self.optimizer = model, optimizer
        self.entropy_mini_loss = MinimizeEntropyLoss()
        #self.batchloss = BatchLoss()
        self.bottleneck_loss = BottleneckLoss()
        self.instance_loss = InstanceLoss()
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(config.lr_scheduler, config.lr,
                                      config.epochs, len(self.train_loader),
                                      config.lr_step, config.warmup_epochs)
        # labels for adversarial training
        #self.source_label = 0
        #self.target_label = 1

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            # cudnn.benchmark = True
            self.model = self.model.cuda()

            #self.D = torch.nn.DataParallel(self.D)
            #patch_replication_callback(self.D)
            #self.D = self.D.cuda()

        self.best_pred_source = 0.0
        self.best_pred_target = 0.0
        self.bn_loss = 100
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
        train_loss, seg_loss_sum, bn_loss_sum, entropy_loss_sum, adv_loss_sum, d_loss_sum, ins_loss_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        target_train_iterator = iter(self.target_train_loader)
        for i, sample in enumerate(tbar):
            itr = epoch * len(self.train_loader) + i
            if self.visdom:
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

            B_image, B_target, B_image_pair = target_sample['image'], target_sample['label'], target_sample['image_pair']

            if self.args.cuda:
                A_image, A_target = A_image.cuda(), A_target.cuda()
                B_image, B_target, B_image_pair = B_image.cuda(), B_target.cuda(), B_image_pair.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred_source, self.best_pred_target)
            #self.scheduler(self.D_optimizer, i, epoch, self.best_pred_source, self.best_pred_target)

            A_output, A_feat, A_low_feat = self.model(A_image)
            B_output, B_feat, B_low_feat = self.model(B_image)
            B_output_pair, B_feat_pair, B_low_feat_pair = self.model(B_image_pair)
            B_output_pair, B_feat_pair, B_low_feat_pair = flip(B_output_pair, dim=-1), flip(B_feat_pair, dim=-1), flip(B_low_feat_pair, dim=-1)

            self.optimizer.zero_grad()
            #self.D_optimizer.zero_grad()

            # Train seg network
            #for param in self.D.parameters():
            #    param.requires_grad = False

            # Supervised loss
            seg_loss = self.criterion(A_output, A_target)
            ins_loss = 0.1 * self.instance_loss(B_output, B_output_pair)
            # Unsupervised bn loss
            main_loss = seg_loss + ins_loss
            main_loss.backward()
            # Train adversarial loss
            #D_out = self.D(prob_2_entropy(F.softmax(B_output)))
            #adv_loss = bce_loss(D_out, self.source_label)
            #main_loss += self.config.lambda_adv * adv_loss
            #main_loss.backward()

            # Train discriminator
            #for param in self.D.parameters():
            #    param.requires_grad = True
            #A_output_detach = A_output.detach()
            #B_output_detach = B_output.detach()
            # source
            #D_source = self.D(prob_2_entropy(F.softmax(A_output_detach)))
            #source_loss = bce_loss(D_source, self.source_label)
            #source_loss = source_loss / 2
            # target
            #D_target = self.D(prob_2_entropy(F.softmax(B_output_detach)))
            #target_loss = bce_loss(D_target, self.target_label)
            #target_loss = target_loss / 2
            #d_loss = source_loss + target_loss
            #d_loss.backward()

            self.optimizer.step()
            #self.D_optimizer.step()

            seg_loss_sum += seg_loss.item()
            ins_loss_sum += ins_loss.item()
            #bn_loss_sum += bottleneck_loss.item()
            #adv_loss_sum += self.config.lambda_adv * adv_loss.item()
            #d_loss_sum += d_loss.item()

            #train_loss += seg_loss.item() + self.config.lambda_adv * adv_loss.item()
            train_loss += seg_loss.item() + ins_loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config.batch_size + A_image.data.shape[0]))
        #print('Loss: %.3f' % train_loss)
        print('Seg Loss: %.3f' % seg_loss_sum)
        print('Ins Loss: %.3f' % ins_loss_sum)
        #print('BN Loss: %.3f' % bn_loss_sum)
        #print('Adv Loss: %.3f' % adv_loss_sum)
        #print('Discriminator Loss: %.3f' % d_loss_sum)

        if self.visdom:
            self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([seg_loss_sum]), win='train_loss', name='Seg_loss',
                          opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                          update='append' if epoch > 0 else None)
            self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([ins_loss_sum]), win='train_loss', name='Ins_loss',
                          opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
                          update='append' if epoch > 0 else None)
            #self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([bn_loss_sum]), win='train_loss', name='BN_loss',
            #              opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
            #              update='append' if epoch > 0 else None)
            #self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([adv_loss_sum]), win='train_loss', name='Adv_loss',
            #              opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
            #              update='append' if epoch > 0 else None)
            #self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([d_loss_sum]), win='train_loss', name='Dis_loss',
            #              opts=dict(title='loss', xlabel='epoch', ylabel='loss'),
            #              update='append' if epoch > 0 else None)

    def validation(self, epoch):
        def get_metrics(tbar, if_source=False):
            self.evaluator.reset()
            test_loss = 0.0
            feat_mean, low_feat_mean, feat_var, low_feat_var = 0, 0, 0, 0
            adv_loss = 0.0
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']

                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                with torch.no_grad():
                    output, low_feat, feat = self.model(image)

                low_feat = low_feat.cpu().numpy()
                feat = feat.cpu().numpy()

                if isinstance(feat, np.ndarray):
                    feat_mean += feat.mean(axis=0).mean(axis=1).mean(axis=1)
                    low_feat_mean += low_feat.mean(axis=0).mean(axis=1).mean(axis=1)
                    feat_var += feat.var(axis=0).var(axis=1).var(axis=1)
                    low_feat_var += low_feat.var(axis=0).var(axis=1).var(axis=1)
                else:
                    feat_mean = feat.mean(axis=0).mean(axis=1).mean(axis=1)
                    low_feat_mean = low_feat.mean(axis=0).mean(axis=1).mean(axis=1)
                    feat_var = feat.var(axis=0).var(axis=1).var(axis=1)
                    low_feat_var = low_feat.var(axis=0).var(axis=1).var(axis=1)

                #d_output = self.D(prob_2_entropy(F.softmax(output)))
                #adv_loss += bce_loss(d_output, self.source_label).item()
                loss = self.criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()

                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)

                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)

            feat_mean /= (i+1)
            low_feat_mean /= (i+1)
            feat_var /= (i+1)
            low_feat_var /= (i+1)
            adv_loss /= (i+1)
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
            #print('Adv Loss: %.3f' % adv_loss)

            # Draw Visdom
            if if_source:
                names = ['source', 'source_acc', 'source_IoU', 'source_mIoU']
            else:
                names = ['target', 'target_acc', 'target_IoU', 'target_mIoU']

            if self.visdom:
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([test_loss]), win='val_loss', name=names[0],
                              update='append')
            #    self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([adv_loss]), win='val_loss', name='adv_loss',
            #                  update='append')
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([Acc]), win='metrics', name=names[1],
                              opts=dict(title='metrics', xlabel='epoch', ylabel='performance'),
                              update='append' if epoch > 0 else None)
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([IoU]), win='metrics', name=names[2],
                              update='append')
                self.vis.line(X=torch.tensor([epoch]), Y=torch.tensor([mIoU]), win='metrics', name=names[3],
                              update='append')

            return Acc, IoU, mIoU, feat_mean, low_feat_mean, feat_var, low_feat_var, adv_loss

        self.model.eval()
        tbar_source = tqdm(self.val_loader, desc='\r')
        tbar_target = tqdm(self.target_val_loader, desc='\r')
        s_acc, s_iou, s_miou, s_m, s_lm, s_v, s_lv, s_adv = get_metrics(tbar_source, True)
        t_acc, t_iou, t_miou, t_m, t_lm, t_v, t_lv, t_adv = get_metrics(tbar_target, False)

        new_pred_source = s_iou
        new_pred_target = t_iou

        bn_loss = np.abs(s_m - t_m).mean() + np.abs(s_lm - t_lm).mean() + np.abs(s_v - t_v).mean() + np.abs(s_lv - t_lv).mean()
        bn_loss = bn_loss.astype('float64')
        #if new_pred_source > self.best_pred_source or new_pred_target > self.best_pred_target:
        if new_pred_source > self.best_pred_source or bn_loss < self.bn_loss:
            is_best = True
            self.best_pred_source = max(new_pred_source, self.best_pred_source)
            #self.best_pred_target = max(new_pred_target, self.best_pred_target)
            self.bn_loss = min(bn_loss, self.bn_loss)
            print('Saving state, epoch:', epoch)
            torch.save(self.model.module.state_dict(), self.args.save_folder + 'models/'
                       + 'epoch' + str(epoch) + '.pth')
        loss_file = {'s_Acc': s_acc, 's_IoU': s_iou, 's_mIoU': s_miou, 't_Acc':t_acc, 't_IoU':t_iou, 't_mIoU':t_miou,
                'bn_loss': bn_loss, 's_adv': s_adv, 't_adv': t_adv}
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
