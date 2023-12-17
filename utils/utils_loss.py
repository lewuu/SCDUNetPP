import torch
import torch.nn.functional as F
from utils.loss.lovasz import LovaszSoftmaxLoss

#=======Loss=====#

def Tversky_loss(inputs, target, alpha = 0.3, beta = 0.7, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1) #(1,65536,2) #softmax多分类?  
    temp_target = target.view(n, -1, ct) 

    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    tversky = (tp + smooth)/(tp + alpha*fp + beta*fn + smooth)
    loss = 1-torch.mean(tversky)

    return loss

def build_loss(outputs, labels, onehot_labels, loss_type='tl'):#

    if loss_type == 'tl':
        losm_loss = LovaszSoftmaxLoss(outputs, labels)
        tver_loss = Tversky_loss(outputs, onehot_labels, alpha = 0.3, beta = 0.7)
        loss = losm_loss+tver_loss

    return loss


#=====Index=====#

# loop
def gen_matrix(pred_mask, gt_mask, num_classes = 2):
    
    """ 
    gt_mask(ndarray): shape -> (height, width)
    pred_mask(ndarray):shape -> (height, width)
    num_classes: 
    """
    pred_mask = torch.softmax(pred_mask.transpose(1, 2).transpose(2, 3).contiguous(), dim = -1)
    pred_mask = pred_mask.argmax(axis=-1) 

    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    count = torch.bincount(num_classes * gt_mask[mask].int() + pred_mask[mask], minlength=num_classes ** 2) 

    cf_mtx = count.reshape(num_classes, num_classes)#.cpu().numpy()
    oa, precision, recall, f_score, iou, miou, mcc = matrix2index(cf_mtx)

    return cf_mtx, oa, precision, recall, f_score, iou, miou, mcc

# gen_matrix
def matrix2index(matrix, smooth = 1e-5):#_macro 
    cf_mtx = matrix
    smooth = torch.tensor(smooth, dtype=torch.float32).to(cf_mtx.device)
    ts1 = torch.tensor(1, dtype=torch.float32).to(cf_mtx.device)

    accuracy  = torch.div(torch.sum(torch.diagonal(cf_mtx)), torch.maximum(torch.sum(cf_mtx), ts1))  
    precision = torch.div(torch.diagonal(cf_mtx), torch.maximum(cf_mtx.sum(0), ts1)) 
    recall = torch.div(torch.diagonal(cf_mtx), torch.maximum(cf_mtx.sum(1), ts1))
    f_score = torch.div(2*precision*recall, (precision+recall+smooth))

    miou = torch.div(torch.diag(cf_mtx), (torch.sum(cf_mtx, axis=1) + torch.sum(cf_mtx, axis=0) - torch.diag(cf_mtx)+smooth))
    mcc = torch.div((torch.prod(torch.diagonal(cf_mtx))-torch.prod(torch.diagonal(torch.fliplr(cf_mtx)))+smooth), 
                    (torch.sqrt(torch.prod(cf_mtx.sum(0), dtype=torch.float32)*torch.prod(cf_mtx.sum(1), dtype=torch.float32))+smooth))

    oa = accuracy.item()
    precision = precision[1].item() 
    recall = recall[1].item()
    f_score = f_score[1].item()
    iou = miou[1].item()
    miou = torch.mean(miou).item()
    mcc = mcc.item()

    return oa, precision, recall, f_score, iou, miou, mcc
