import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F
from common_functions import CustumDataset
import copy

def _GetIndex(data_1):
    idx = -1
    if data_1 > 0.5 and data_1 <= 0.6:
        idx = 0
    elif data_1 >0.6 and data_1 <= 0.7:
        idx = 1
    elif data_1 >0.7 and data_1 <= 0.8:
        idx = 2
    elif data_1 >0.8 and data_1 <= 0.9:
        idx = 3
    elif data_1 >0.9 and data_1 <= 1.0:
        idx = 4
    return idx

def _GetIndex_avgfeat(data_1):
    idx = -1
    if data_1 > 0.5 and data_1 <= 0.6:
        idx = 0
    return idx
        
def GetSplitLoaders_BinaryClasses(list_correct, 
                                    dataset, 
                                    train_aug=None,
                                    num_store_per=5,
                                    batch_size=128):

    correct_loader=[[],[]]
    for i in range(num_store_per): # Go through 5 level: 0 -> 4
        list_temp = [list_correct[i][0],list_correct[i][1]]
        for rf in range(len(list_temp)): # in range 2
            if not list_temp[rf]: # no item Real / Fake in this confident interval
                correct_loader[rf].append([])
                continue
            custum = CustumDataset(np.array(dataset.data[list_correct[i][rf]]),
                                   np.array(dataset.target[list_correct[i][rf]]),
                                   train_aug)
            correct_loader[rf].append(DataLoader(custum,
                                     batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
    
    list_length_realfakeloader = [[len(j.dataset) if j else 0 for j in i] for i in correct_loader]
    print("list_length_realfakeloader :", list_length_realfakeloader)
    return correct_loader, np.sum(list_length_realfakeloader)/len(dataset.target)

def GetSplitLoadersRealFake(list_correct,dataset,train_aug=None,num_store_per=5):
    correct_loader=[[],[]]
    for i in range(num_store_per):
        list_temp = [list_correct[i][0],list_correct[i][1]]
        for rf in range(len(list_temp)):
            if not list_temp[rf] :
                correct_loader[rf].append([])
                continue
            temp_dataset = copy.deepcopy(dataset)
            temp_dataset.data = np.array(temp_dataset.data[list_correct[i][rf]])
            temp_dataset.target = np.array(temp_dataset.target[list_correct[i][rf]])
            custum = CustumDataset(temp_dataset.data,temp_dataset.target,train_aug)
            correct_loader[rf].append(DataLoader(custum,
                                     batch_size=200, shuffle=False, num_workers=4, pin_memory=True))    
    
    list_length_realfakeloader = [[len(j.dataset) if j else 0 for j in i] for i in correct_loader]
    return correct_loader,np.array(list_length_realfakeloader)/len(dataset.target)

def GetListTeacherFeatureFakeReal(model, loader, mode='Xception', showScatter = False, device='cuda'):
    list_features = [[],[]] # 2 x 5 x 2048
    maxpool = nn.MaxPool2d(4)
    model.eval()
    with torch.no_grad():
        train_results, labels = [[], []], [[], []]
        for i in range(len(loader)):
            for j in range(len(loader[i])):
                if not loader[i][j] :
                    train_results[i].append([])
                    list_features[i].append(torch.tensor([0]*2048))
                    continue
                temp = None
                for _, (img, label) in enumerate(loader[i][j]):
                    train_results[i].append(model(img.to(device)).cpu().detach().numpy())
                    labels[i].append(label)
                    if mode == 'Efficient':
                        test = model.extract_features(img.to(device))
                    elif mode == 'Xception' or mode == 'MobileNet2':
                        test = model.features(img.to(device))
                    elif mode == 'ResNet' or mode == 'ResNet18':
                        test = model.features(img.to(device))

                    if temp is not None:
                        temp = torch.cat((temp, maxpool(test)))
                    else:
                        temp = maxpool(test)

                
                temp = temp.squeeze((1,2,3)).mean(dim=0)
                list_features[i].append(temp.detach().cpu().numpy())

    return list_features



def func_correct(model, data_loader, device='cuda'):
    """
    Expensive function 
    0: (confidence level 0) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    1: (confidence level 2) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    2: (confidence level 3) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    3: (confidence level 4) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    4: (confidence level 5) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    
    """
    list_correct = [[[],[]] for i in range(5)]
    model.eval()
    all_pred = []
    cnt = 0
    with torch.no_grad():
        pbar = tqdm(total=len(data_loader), ncols=50)
        for i, (inputs, targets) in enumerate(data_loader):
            _inputs = inputs.to(device)
            _targets = targets.to(device)
            outputs = model(_inputs)
            temp = F.softmax(outputs,dim=1)
            all_pred = np.concatenate((all_pred, temp.cpu().numpy()), axis=0) if len(all_pred) else temp.cpu().numpy()
            for l in range(len(_targets)):
                idx = _GetIndex(temp[l][_targets[l]].data)
                if idx >= 0:
                    if _targets[l]==0 : 
                        list_correct[idx][0].append(cnt)
                    else: 
                        list_correct[idx][1].append(cnt)
                cnt += 1
            pbar.update()
        pbar.close()
        return list_correct, all_pred




def func_correct_avgfeat(model, data_loader, device='cuda'):
    list_correct = [[[],[]] for i in range(5)]
    model.eval()
    cnt=0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            _inputs = inputs.to(device)
            _targets = targets.to(device)
            outputs = model(_inputs)
            temp = F.softmax(outputs,dim=1)
            for l in range(len(_targets)):
                idx = _GetIndex_avgfeat(temp[l][_targets[l]].data)

                if idx >= 0:
                    if _targets[l]==0 : 
                        list_correct[idx][0].append(cnt)
                    else : list_correct[idx][1].append(cnt)
                cnt+=1
        return list_correct

def GetRatioData(list_real_fake,correct_cnt):
    if correct_cnt == 0 :
        return 0
    list_length_realfakeloader = np.array([[len(j) if j else 0 for j in i] for i in list_real_fake])
    return list_length_realfakeloader/correct_cnt


def correct_binary_simple(inputs, penul_ft, outputs, targets, device='cuda'):
    """    
    0: (confidence level 0) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    1: (confidence level 1) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    2: (confidence level 2) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    3: (confidence level 3) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
    4: (confidence level 4) [list_real_image_correctly_clss], [list_fake_image_correctly_clss]
   
    """
    maxpool = nn.MaxPool2d(4)
    penul_ft = maxpool(penul_ft).squeeze() # batchsize x 2048
    rep_ft_partitions =  [[[], []] for i in range(5)] # 5 x 2
    _inputs = inputs
    _targets = targets
    temp = nn.Softmax(dim=1)(outputs) # batch_size x 2 
    temp = temp.cpu()
    for l in range(len(_targets)): # run through the batch size
        idx = _GetIndex(temp[l][_targets[l]].data) # get the index of confidence intervals  (0.5, 0.6] -> 0, (0.6, 0.7] -> 2, etc.
        if idx >= 0: # correcly classify
            cls_ = _targets[l]  # 0 / 1
            rep_ft_partitions[idx][cls_].append(penul_ft[l])
               
    return rep_ft_partitions