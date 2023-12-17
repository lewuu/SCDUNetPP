import os
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils_loss import build_loss, gen_matrix, matrix2index
from model._build_model import build_model       
from dataloader import build_dataset
from configs import cfg


if __name__ == '__main__':
 
    model_type   = cfg.model.model_type    
    num_classes  = cfg.model.num_classes
    dataset_path = cfg.dataset.dataset_path
    test_lines   = cfg.dataset.test_lines
    input_shape  = cfg.dataloader.input_shape
    in_channels  = cfg.dataloader.in_channels
    batch_size   = 1
    cuda         = cfg.train.cuda

    with open((test_lines),"r") as f:
        test_lines   = f.readlines()

    test_dataset   = build_dataset(test_lines, input_shape, in_channels, num_classes, False, dataset_path)
    test_loader    = DataLoader(test_dataset, shuffle = False, batch_size=batch_size, num_workers=1, drop_last=False)

    model = build_model(model_type)

    device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join("checkpoints/luding", cfg.model.model_type, cfg.train.ckpt_test)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    print('{} model, and classes loaded.'.format(model_path))

    if cuda:   
        model = model.cuda()

    epoch_step_test  = len(test_lines) // batch_size  
    if len(test_lines) % batch_size != 0:
        epoch_step_test += 1
  
    test_loss = 0
    cf_mtx = torch.zeros((num_classes, num_classes)).cuda()

    t1 = time.time()

    with tqdm(total = epoch_step_test, desc = f'Epoch {0 + 1}/{1}', postfix = dict, mininterval = 0.3) as pbar:
        for iteration, batch in enumerate(test_loader):
            imgs, labels, onehot_labels = batch 
            with torch.no_grad():
                if cuda:
                    imgs          = imgs.cuda()
                    labels        = labels.cuda()
                    onehot_labels = onehot_labels.cuda()

                outputs = model(imgs)
                loss = build_loss(outputs, labels, onehot_labels)
                test_loss += loss.item()

                test_cf_mtx = gen_matrix(outputs, labels)[0] 
                cf_mtx = cf_mtx + test_cf_mtx

            pbar.update(1)

    t2 = time.time()
    FPS = (iteration+1) / (t2 - t1)
    print('FPS:       ', FPS)

    decimal_digits = 10000

    test_acc, test_precision, test_recall, test_f_score, test_iou, test_miou, test_mcc = matrix2index(cf_mtx)
    
    test_acc       = format(test_acc, '.4f')
    test_precision = format(test_precision, '.4f')
    test_recall    = format(test_recall, '.4f')
    test_f_score   = format(test_f_score, '.4f')
    test_iou       = format(test_iou, '.4f') 
    test_miou      = format(test_miou, '.4f') 
    test_mcc       = format(test_mcc, '.4f') 

    print('loss:      ', test_loss / (iteration + 1))
    print('acc:       ', test_acc)
    print('precision: ', test_precision)
    print('recall:    ', test_recall)
    print('f1:        ', test_f_score)
    print('iou:       ', test_iou)
    print('miou:      ', test_miou)
    print('mcc:       ', test_mcc)
