import torch
import torch.optim as optim
import random
import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn
from configs import cfg
from dataloader import build_dataset
from model._build_model import build_model
from utils.utils_loop import loop_one_epoch


def build_optimizer(cfg, model, optim_name):
    parameters = model.parameters()
    optimizer = None
    base_lr=cfg.optimizer.base_lr

    if optim_name == 'adam':
        optimizer = optim.Adam(parameters, base_lr, 
                               weight_decay = cfg.optimizer.weight_decay)
    elif optim_name == 'sgd':
        optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True,
                              lr=base_lr, weight_decay=cfg.optimizer.weight_decay)
    elif optim_name == 'adamw':
        optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=base_lr, weight_decay=cfg.optimizer.weight_decay)

    return optimizer

def build_scheduler(cfg, optimizer, lr_scheduler):
    if lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                T_max=cfg.train.total_epoch, 
                eta_min=cfg.optimizer.min_lr)#1e-5

    return lr_scheduler

def weights_init(model, init_method = 'normal', init_gain = 0.02):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            if init_method   == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_method == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_method == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_method)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    print('initialize network with %s method' % init_method)
    model.apply(init_func)

def set_random_seed(seed, deterministic=False):

    random.seed(seed)             
    np.random.seed(seed)       
    torch.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed) 
    torch.cuda.manual_seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed) 
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    set_random_seed(0, deterministic=True)
    
    ##   load cfgs
    model_type   = cfg.model.model_type
    num_classes  = cfg.model.num_classes
    cls_weights  = torch.tensor([0.5,0.5], dtype=torch.float32)
    # DATASET
    dataset_path = cfg.dataset.dataset_path
    train_lines  = cfg.dataset.train_lines
    val_lines    = cfg.dataset.val_lines
    # DATALOADER
    isAug        = cfg.dataloader.isOnLineAug
    shuffle      = cfg.dataloader.isShuffle
    batch_size   = cfg.dataloader.batch_size
    num_workers  = cfg.dataloader.num_workers
    input_shape  = cfg.dataloader.input_shape
    in_channels  = cfg.dataloader.in_channels
    # TRAIN
    cuda         = cfg.train.cuda
    end_epoch    = cfg.train.total_epoch
    resume_path  = cfg.train.ckpt_resume 
    ckpt_savpath = cfg.train.ckpt_savepath
    freeze_param = cfg.train.freeze_param

    time_str = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    save_path = os.path.join(ckpt_savpath, "loss_" + str(time_str)) 
    os.makedirs(save_path)

    ##   read txt
    with open((train_lines),"r") as f:
        train_lines = f.readlines()
        random.shuffle(train_lines)
    with open((val_lines),"r") as f:
        val_lines   = f.readlines()

    ##   define data preprocessing
    train_data = build_dataset(train_lines, input_shape, in_channels, num_classes, isAug, dataset_path)
    val_data = build_dataset(val_lines, input_shape, in_channels, num_classes, False, dataset_path)

    ##   load data
    train_loader = DataLoader(train_data, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, 
                              pin_memory=True, drop_last=True,persistent_workers=True)
    val_loader = DataLoader(val_data, shuffle = False, batch_size = batch_size, num_workers = num_workers, 
                            pin_memory=True, drop_last=True,persistent_workers=True)

    ##   load model
    model = build_model(model_type)

    ##   init model if not pretained
    if not cfg.train.pretrain:
        weights_init(model, init_method='normal')#kaiming

    if cfg.train.cuda:
        model.to(device)

    ##   build optimizer
    optimizer = build_optimizer(cfg, model, 'adamw')
    lr_scheduler = build_scheduler(cfg, optimizer, 'cosine')

    ##   resume or not
    start_epoch = 0
    if cfg.train.resume:
        checkpoint  = torch.load(resume_path, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    ##   loop
    step_train = len(train_lines) // batch_size
    step_val = len(val_lines) // batch_size

    if freeze_param:
        checkpoint  = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.freeze_param()

    for epoch in range(start_epoch, end_epoch):
        loop_one_epoch(model, save_path, optimizer,lr_scheduler, epoch, 
                end_epoch, step_train, step_val, train_loader, val_loader,
                cuda, cls_weights, num_classes)  

        lr_scheduler.step() 


if __name__ == '__main__':

    main()