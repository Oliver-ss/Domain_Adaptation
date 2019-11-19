'''*************************************************************************
	> File Name: common.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 21 Oct 2019 01:46:28 PM EDT
 ************************************************************************'''
class Config:
    #Model
    backbone = 'mobilenet' #backbone name ['resnet', 'xception', 'drn', 'mobilenet']
    out_stride = 8 #network output stride

    #Data
    all_dataset = ['Shanghai', 'Vegas', 'Paris', 'Khartoum']
    dataset = 'Paris'
    train_num_workers = 4
    finetune_num_workers = 0
    val_num_workers = 2
    img_root = '/usr/xtmp/satellite/spacenet/'
    #Train
    batch_size = 16
    finetune_batch_size = 2
    freeze_bn = False
    sync_bn = False
    loss = 'ce' #['ce', 'focal']
    epochs = 5000
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    lr_scheduler = 'cos'
    lr_step = 5
    warmup_epochs = 10
    beta = 1
    # T = 100 # how many epoch in a period


config = Config()
