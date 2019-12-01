# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:40:10 2019

@author: Chanakya-vc
"""
import torch
from  torchvision import transforms, datasets
from numpy_patch_optimization_vcc import convex_update
import numpy as np
from tri_loss.model.Model import Model
from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.model.loss import global_loss
#from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.utils.utils import load_state_dict
from datacollector_domain import datacollector
import os
from PIL import Image
from evaluatePRID import evaluate 
#from torch.utils import data
from torch.utils.data import Dataset, DataLoader

class Prid_Dataset(Dataset):
  def __init__(self,data_src,data_src_cam_a,data_src_cam_b):
      self.img_dict={}
      path_cam_a,list_cam_a=datacollector(data_src_cam_a,['cam_a'])
      path_cam_b,list_cam_b=datacollector(data_src_cam_b,['cam_b'])
      path,list_cameras=datacollector(data_src)
      for i in range(100):
          init_a=sum(list_cam_a[0:i])
          end_a=list_cam_a[i]-1
          paths_index_cam_a=path_cam_a[init_a:end_a]
          
          init_b=sum(list_cam_b[0:i])
          end_b=list_cam_b[i]-1
          paths_index_cam_b=path_cam_b[init_b:end_b]
          imgs=[]
          
          for img_path in paths_index_cam_a:
              img=Image.open(img_path)
              imgs.append(img)
          for img_path in paths_index_cam_b:
              img=Image.open(img_path)
              imgs.append(img)
          imgs=np.concatenate(imgs)
          
          self.img_dict[i]=imgs  

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_dict)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        imgs = self.imgs_dict[index]

        # Load data and get label
        X = torch.from_numpy(imgs)
        noise=np.random.randint(1,10)
        if(noise==1):
            y=np.random.randint(1,100)
        else:
            y = index

        return X, y




data_src=r"/home/anurags/Dataset/prid/multi_shot/"
data_src_cam_a=r"/home/anurags/Dataset/prid/multi_shot/cam_a"
data_src_cam_b=r"/home/anurags/Dataset/prid/multi_shot/cam_b"
PATH=r"/home/anurags/weights/prid/results/multi_shot/model_weight.pth"
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
    model_weight=torch.load('/home/anurags/weights/market1501/stride1/model_weight.pth')
    load_state_dict(model,model_weight)
    model.cuda() #convert model to GPU type
    #Batch size should be gretater than the vlaue of epsilon
    parameters={'alpha': 0.0001,'beta':0.1,'gamma':1000,'lambda':1,'epsilon':16,'eta':0,'iterations':3}
    batch_size=15
    epochs=20
    tri_loss = TripletLoss(margin=0.7)

#    img_mapping=datacollector(data_src)
#    img_mapping_a_to_b = datacollector(data_src, ['cam_a'])
#    img_mapping_b_to_a = datacollector(data_src, ['cam_b'])
#    print("--------- image mapping is over ------------")
#    Z = Q = A = Y2 = np.zeros((len(img_mapping),len(img_mapping)))
#    E = Y1 = np.zeros((2048,len(img_mapping)))
#    tensor_dict={'Z':Z,'Q':Q,'A':A,'E':E,'Y1':Y1,'Y2':Y2,'mu':0.0001} 
    
# =============================================================================
#     Load Dataset and apply transform
# =============================================================================
    prid=Prid_Dataset(data_src,data_src_cam_a,data_src_cam_b,transform=data_transform)
    
    dataloader=DataLoader(prid,batch_size=4,shuffle=True,num_workers=4)
    
    
    
    for i in range(epochs):
        model.eval()
        model.zero_grad()
        torch.cuda.empty_cache()       
# =============================================================================
# # Read images from the mapping and pass them through the model to obtain features 
# =============================================================================   
        
        for imgs ,labels in dataloader:
            imgs, labels= imgs.to(device),labels.to(device)
            feat=model(imgs)
            loss, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(tri_loss, feat, labels)
            loss.backwards()

       