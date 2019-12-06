import data.spacenet
from torch.utils.data import DataLoader

def make_data_loader(config):

    train_set = spacenet.Spacenet(city=config.dataset, split='train', img_root=config.img_root)
    val_set = spacenet.Spacenet(city=config.dataset, split='val', img_root=config.img_root)
    #test_set = spacenet.Spacenet(city=config.dataset, split='test', img_root=config.img_root)

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
            num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
            shuffle=False, num_workers=config.val_num_workers, drop_last=True)
    #test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.val_num_workers)
    test_loader = None

    return train_loader, val_loader, test_loader, num_class

def make_target_data_loader(config):

    train_set = spacenet.Spacenet(city=config.target, split='train', img_root=config.img_root, if_pair=True)
    val_set = spacenet.Spacenet(city=config.target, split='val', img_root=config.img_root, if_pair=True)
    #test_set = spacenet.Spacenet(city=config.dataset, split='test', img_root=config.img_root)

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
            num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
            num_workers=config.val_num_workers, drop_last=True)
    #test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.val_num_workers)
    test_loader = None

    return train_loader, val_loader, test_loader, num_class
