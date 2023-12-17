import torch
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.utils_loss import build_loss, gen_matrix, matrix2index


def loop_one_epoch(model, save_path, optimizer, lr_scheduler, epoch, end_epoch, 
        step_train, step_val, train_loader, val_loader, cuda, weights, num_classes):

    for param_group in optimizer.param_groups:
        epoch_lr = param_group['lr']
    train_loss = 0
    val_loss = 0

    scaler = GradScaler(enabled=True) 

    print('Start Epoch')
    
    """ Train """
    model.train()
    with tqdm(total = step_train, desc = f'Train {epoch+1}/{end_epoch}', 
              postfix = dict, mininterval = 0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= step_train: 
                break

            optimizer.zero_grad() 
            with autocast(enabled=True): 
                imgs, labels, onehot_labels = batch 
                with torch.no_grad():
                    if cuda:
                        imgs = imgs.cuda() 
                        labels = labels.cuda()
                        onehot_labels = onehot_labels.cuda()
                        weights = weights.cuda()

                outputs = model(imgs)
                loss = build_loss(outputs, labels, onehot_labels)

            scaler.scale(loss).backward() 
            scaler.step(optimizer)    
            scaler.update()     

            train_loss += loss.item()
        
            pbar.set_postfix(**{'tra_loss': train_loss / (iteration + 1),
                                'lr'      : epoch_lr})
            pbar.update(1)


    """ Validation """
    model.eval()
    cf_mtx = torch.zeros((num_classes, num_classes)).cuda()
    with tqdm(total = step_val, desc = f'Valid {epoch+1}/{end_epoch}', 
              postfix = dict, mininterval = 0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= step_val:
                break
            imgs, labels, onehot_labels = batch
            with torch.no_grad():
                if cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                    onehot_labels = onehot_labels.cuda()
                    weights = weights.cuda()

                outputs = model(imgs)

                loss = build_loss(outputs, labels, onehot_labels)
                val_loss += loss.item()

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                test_cf_mtx = gen_matrix(outputs, labels, num_classes)[0] 
                cf_mtx = cf_mtx + test_cf_mtx
              
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1)})
            pbar.update(1)

    f_score, iou = matrix2index(cf_mtx)[3:5]

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch+1,
    }

    torch.save(checkpoint, os.path.join(save_path, 'epc%03d-trloss%.3f-valoss%.3f-iou%.3f-f1-%.3f.pth'%(
        epoch+1, train_loss/step_train, val_loss/step_val, iou, f_score)))
                
    print('Finish Epoch')

    print('Epoch: '+ str(epoch+1) + '/' + str(end_epoch) + 
          ' >> train loss: %.3f || val loss: %.3f || val iou: %.3f || val f1: %.3f' 
          % (train_loss / step_train, val_loss / step_val, iou, f_score))
