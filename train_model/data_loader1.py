import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F

from skimage import transform
from PIL import Image
import json
import cv2
import struct
import glob
from STRANet_utils import *
import time, random

class ImageFolder1(data.Dataset):

    def __init__(self,mode,batch_size,cuSize,debug):
        self.debug=debug  #train or debug
        self.mode=mode
        self.cuSize=cuSize
        self.list=self.getlist()
        self.batch_size=batch_size        
        self.max_pool=torch.nn.MaxPool2d((2,2))
        
    def __getitem__(self, index):
        tmp={}
        w, h = map(int, self.list[index]['path'].split('_')[-2].split('x'))
        tmp['top']=self.list[index]['top']
        tmp['left']=self.list[index]['left']
        tmp['hgt']=self.list[index]['hgt']
        tmp['wid']=self.list[index]['wid']
        tmp['gt']=self.list[index]['gt']
        tmp['qp']=self.list[index]['qp']
        if self.cuSize!=5:
            y = import_yuv(self.list[index]['path'], h, w, 1, yuv_type='420p', start_frm=0, only_y=True)[0] # (h, w)
            tmp['image']=torch.unsqueeze( \
            torch.from_numpy(y[tmp['top']:tmp['top']+tmp['hgt'], tmp['left']:tmp['left']+tmp['wid']]), dim=0)
        else:
            y, u, v = import_yuv(self.list[index]['path'], h, w, 1, yuv_type='420p', start_frm=0, only_y=False) # (h, w)
            y=torch.from_numpy(y[:,tmp['top']:tmp['top']+tmp['hgt']*2, tmp['left']:tmp['left']+tmp['wid']*2]).float()
            y=self.max_pool(y)
            u=torch.from_numpy(u[:,tmp['top']//2:tmp['top']//2+tmp['hgt'], tmp['left']//2:tmp['left']//2+tmp['wid']]).float()
            v=torch.from_numpy(v[:,tmp['top']//2:tmp['top']//2+tmp['hgt'], tmp['left']//2:tmp['left']//2+tmp['wid']]).float()
            yuv_list=[y,u,v]
            tmp['image']=torch.cat(yuv_list,dim=0)


        def rotate_values():
            new_gt=tmp['gt'].clone()
            new_gt[2], new_gt[3], new_gt[4], new_gt[5]=tmp['gt'][3], tmp['gt'][2], tmp['gt'][5], tmp['gt'][4]
            tmp['gt']=new_gt

        if tmp['hgt']>tmp['wid']:
            tmp['image']=tmp['image'].transpose(2,1)
            rotate_values()
            tmp['hgt']=self.list[index]['wid']
            tmp['wid']=self.list[index]['hgt']
        
        flipped=tmp['image'].clone()

        rand=random.random()
        tmp['random']=rand

        if self.mode=='train':
            if (rand<0.125 and self.cuSize!=2) or (rand<0.25 and self.cuSize==2): 
                for i in range(tmp['hgt']):
                    flipped[:,tmp['hgt']-1-i]=tmp['image'][:,i]
            elif (rand<0.25 and self.cuSize!=2) or (rand<0.5 and self.cuSize==2):
                for i in range(tmp['wid']):
                    flipped[:,:,tmp['wid']-1-i]=tmp['image'][:,:,i]
            elif (rand<0.375 and self.cuSize!=2) or (rand<0.75 and self.cuSize==2):
                for i in range(tmp['wid']):
                    flipped[:,:,tmp['wid']-1-i]=tmp['image'][:,:,i]
                for i in range(tmp['hgt']):
                    tmp['image'][:,tmp['hgt']-1-i]=flipped[:,i]
                tmp['image']=tmp['image'].float()
                return tmp
            elif (rand>0.5 and self.cuSize<=1):
                rotated=tmp['image'].clone()
                rotate_values()
                if self.cuSize==1:
                    if tmp['qp']%4==3 or tmp['qp']%4==2:
                        tmp['qp']=tmp['qp']-tmp['qp']%4+(5-tmp['qp']%4)

                for i in range(tmp['hgt']):
                    rotated[:,i]=tmp['image'][:,:,i]
                if rand<0.625:
                    for i in range(tmp['hgt']):
                        flipped[:,tmp['hgt']-1-i]=rotated[:,i]
                elif rand<0.75:
                    for i in range(tmp['wid']):
                        flipped[:,:,tmp['wid']-1-i]=rotated[:,:,i]
                elif rand<0.875:
                    for i in range(tmp['wid']):
                        flipped[:,:,tmp['wid']-1-i]=rotated[:,:,i]
                    for i in range(tmp['hgt']):
                        tmp['image'][:,tmp['hgt']-1-i]=flipped[:,i]
                    tmp['image']=tmp['image'].float()
                    return tmp
                else:
                    flipped=rotated

        tmp['image']=flipped.float()
        
        return tmp
    def getlist(self):
        datalist=[]
        def gen_datalist(qp,train_or_test):
            self.yuv_path_list=[]
            name_list=glob.glob("../collected_"+str(self.cuSize)+'/'+str(qp)+"/*")
            self.yuv_list = sorted(name_list, key=lambda x:int(os.path.basename(x).split('_')[0]), reverse=False)
            random.seed(65345)

            if self.cuSize==2:
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *9//10) #24//100
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index)) #//200
            elif self.cuSize==1 or self.cuSize==3:
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *9//10)
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index)) #//500
            elif self.cuSize==4:
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *9//10)
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index))
            else: #cuSize==0 or 5
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *9//10)
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index))
            
            if train_or_test=='train':
                self.yuv_path_list = np.array(self.yuv_list)[train_list_index]
            else:
                self.yuv_path_list = np.array(self.yuv_list)[val_list_index]

            for i,path in enumerate(self.yuv_path_list):
                if self.debug=='debug':
                    if i==50:
                        print("break")
                        break
                with open(path,"r") as write_file:
                    cu_pic=json.load(write_file)

                if self.cuSize%5==0:
                    mode_num=1
                elif self.cuSize>=1 and self.cuSize<=3:
                    mode_num=4
                elif self.cuSize==4:
                    mode_num=6

                for mode in range(mode_num):
                    for key,splits in cu_pic['prob'][mode].items():
                        data_item={}
                        data_item['qp']=(qp-22)//5

                        if self.cuSize==1:
                            data_item['qp']=mode+data_item['qp']*4
                        elif self.cuSize==2 or self.cuSize==3:
                            data_item['qp']=mode%2+data_item['qp']*2
                        elif self.cuSize==4:
                            data_item['qp']=mode%3+data_item['qp']*3

                        data_item['path']="../../saved_from_server/run-10.23/yuv/0/"+path.split("/")[-1].split('.')[0]+".yuv"

                        gt=torch.zeros((6))
                        sum_prob=0
                        for idx,par_mode in enumerate(splits):
                            gt[idx]=par_mode    
                            sum_prob+=par_mode
                        
                        gt/=sum_prob
                        data_item['gt']=gt
                        
                        if self.cuSize==0 or self.cuSize==5:
                            data_item['hgt']=32
                            data_item['wid']=32
                        elif self.cuSize==1:
                            data_item['hgt']=16
                            data_item['wid']=16
                        elif self.cuSize==2:
                            if mode//2==0:
                                data_item['hgt']=16
                                data_item['wid']=32
                            else:
                                data_item['hgt']=32
                                data_item['wid']=16
                        elif self.cuSize==3:
                            if mode//2==0:
                                data_item['hgt']=8
                                data_item['wid']=32
                            else:
                                data_item['hgt']=32
                                data_item['wid']=8
                        elif self.cuSize==4:
                            if mode//3==0:
                                data_item['hgt']=8
                                data_item['wid']=16
                            else:
                                data_item['hgt']=16
                                data_item['wid']=8

                        data_item['top']=int(key.split("_")[0])
                        data_item['left']=int(key.split("_")[1])
                        datalist.append(data_item)
                            
        if self.mode=='train':
            start=time.time()
            gen_datalist(37,'train')
            gen_datalist(32,'train')
            gen_datalist(27,'train')
            gen_datalist(22,'train')
            print('Loading dataset times is {}: '.format(time.time() - start))
        else:
            gen_datalist(37,'test')
            gen_datalist(32,'test')
            gen_datalist(27,'test')
            gen_datalist(22,'test')

        print("getting list finished")
        return datalist
    def __len__(self):
        return len(self.list)-len(self.list)%self.batch_size

def get_loader(cuSize,batch_size, num_workers=2, mode='train',debug='train'):
    """Builds and returns Dataloader."""
    dataset = ImageFolder1(mode=mode,batch_size=batch_size,cuSize=cuSize,debug=debug)
    if debug=='train':
        return data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
    else:
        return data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
