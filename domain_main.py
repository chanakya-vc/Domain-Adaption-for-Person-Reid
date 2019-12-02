# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:40:10 2019

@author: Chanakya-vc
"""
import random
import torch
from  torchvision import transforms, datasets
import numpy as np
from tri_loss.model.Model import Model
from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.model.loss import global_loss
from tri_loss.utils.utils import load_state_dict
from datacollector import *
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class Prid_Dataset(Dataset):
    def __init__(self,data_src,transform,train):
        self.transform=transform
        self.train=train
        self.range=(0,100) if self.train else (101,200)
        self.ImgMapping=datacollector(data_src,['cam_a','cam_b'],self.range)
        self.IdSet,self.IdtoImgMappingDict=idcollector(self.ImgMapping)
        self.IdSet=list(self.IdSet)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.IdSet)

    def __getitem__(self, index):
        'Generates one sample of data gives you all the samples of an id'
        # Select sample
        img_paths = self.IdtoImgMappingDict[index]
        if len(img_paths)>20:
            img_paths=random.sample(img_paths,20)
        X=[]
        # Load data and get label
        for path in img_paths:
            img=Image.open(path)
            X.append(self.transform(img).numpy())
        X = np.array(X)
        y = np.zeros((X.shape[0],))
        noise=np.random.randint(1,10)
        for i in range(X.shape[0]):
            y[i]=np.random.randint(self.range[0],self.range[1]) if noise==1 else index
        return torch.tensor(X), torch.tensor(y)


def my_collate(batch):
    data = []
    target = []
    for sub_batch in batch:
        for img in sub_batch[0]:
            data.append(img)
        for label in sub_batch[1]:
            target.append(label)
    target = torch.LongTensor(target)
    return torch.stack(data), target

data_src=r"/home/iacvlab/Dataset/prid/multi_shot/"
PATH=r"/home/iacvlab/anurag/ccis/weights/market1501_stride1/model_weight.pth"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#load data
data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.486, 0.459, 0.408],
                             std=[0.229, 0.224, 0.225])
        ])

if __name__ == '__main__':
    #load the model
    model=Model(last_conv_stride=1)
    model_weight=torch.load(PATH)
    print(model)
    model.load_state_dict(model_weight)
    model.cuda() 
    batch_size=4
    epochs=20
    tri_loss = TripletLoss(margin=0.7)

#    img_mapping=datacollector(data_src)
#    img_mapping_a_to_b = datacollector(data_src, ['cam_a'])
#    img_mapping_b_to_a = datacollector(data_src, ['cam_b'])
#    Load Dataset and apply transform

    prid=Prid_Dataset(data_src,transform=data_transform,train=True)
    
    dataloader=DataLoader(prid,batch_size=batch_size,collate_fn=my_collate,shuffle=True,num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4,weight_decay=0.0005)    
    
    for i in range(epochs):
        model.zero_grad()
        torch.cuda.empty_cache()       
# =============================================================================
# # Read images from the mapping and pass them through the model to obtain features 
# =============================================================================           
        for imgs, labels in dataloader:
            imgs, labels= imgs.to(device),labels.to(device)
            print(imgs.shape,labels.shape)
            feat=model(imgs)
            loss, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(tri_loss, feat, labels)
            loss.backward()
            # run adam for each batch
            optimizer.step()

       
