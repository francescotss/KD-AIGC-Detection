import os
import torch
import shutil
import numpy as np
import os

def _seed_everything(random_seed=41):
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        
def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar' , AUC_BEST = False, ACC_BEST = False):
    name_save = ''
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if AUC_BEST :
        name_save = 'model_best.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, name_save))
    if ACC_BEST :
        name_save = 'model_best_accuracy.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, name_save))
def save_checkpoint_for_unlearning(state, checkpoint='checkpoint', filename='checkpoint.pth.tar' , isLoss=False, isAcc=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if isLoss :
        name_save = 'model_lowest_loss.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, name_save))
    if isAcc :
        name_save = 'model_best_acc.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, name_save))

def adjust_learning_rate(optimizer, epoch, opt):
    lr_set = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    lr_list = opt.schedule.copy()
    lr_list.append(epoch)
    lr_list.sort()
    idx = lr_list.index(epoch)
    opt.lr *= lr_set[idx]
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr
        
#mhkim add-temp
def save_arr_acc_loss(list_acc,list_real_acc,list_fake_acc,list_loss,
                      list_val_acc,list_val_real_acc,list_val_fake_acc,list_val_loss,
                      path):
    list_final,list_val_final=[],[]
    
    list_final.append(list_acc)
    list_final.append(list_real_acc)
    list_final.append(list_fake_acc)
    list_final.append(list_loss)
    list_val_final.append(list_val_acc)
    list_val_final.append(list_val_real_acc)
    list_val_final.append(list_val_fake_acc)
    list_val_final.append(list_val_loss)
    train_dir = path + '_train'
    val_dir = path + '_val'

    np.save(train_dir,list_final)
    np.save(val_dir,list_val_final)

def loss_clampping(loss, min_val, max_val):
    if loss != 0.0 and (torch.isinf(loss) or torch.isnan(loss)):
        loss = 0.0 
    if loss > 0.0:
        loss = torch.clamp(loss, min=min_val, max=max_val)
    return loss

def ReduceWeightOnPlateau(weight, factor=0.8):
    """
        Reduce the conservative of the CL model
    """
    weight *= factor
    return weight