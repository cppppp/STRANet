from ntpath import join
import os
import numpy as np
import time
from scipy.fft import ifftn
import torch
from torch import optim
from data_loader1 import get_loader
torch.backends.cudnn.deterministic = True
from tqdm import tqdm
import torch.distributed as dist
import gc
from itertools import chain
from torch.optim.lr_scheduler import ReduceLROnPlateau
from new_stf import Win_noShift_Attention
from network import *

from matplotlib import pyplot as plt

output=[0,0,0,0,0,0]

class Solver(object):
    def __init__(self, config, gpus):
        self.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
        self.cuSize=config.cuSize
        self.isTrain=config.isTrain
        self.isDebug=config.isDebug

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.num_workers=config.num_workers

        # Path
        self.model_path = config.model_path
        self.gpus = gpus
        self.build_model()
        #self.load_model()

    def load_model(self):
        def my_load_state_dict(module,state_dict):
            from collections import OrderedDict
            if 'module.' in list(state_dict.keys())[0]:  # multi-gpu training
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove module
                    new_state_dict[name] = v
                module.load_state_dict(new_state_dict, strict = True)
            else:
                module.load_state_dict(state_dict, strict=False)#, strict = True)

        for i,module in enumerate(self.cuSize_list):
            my_load_state_dict(module,torch.load(os.path.join(self.model_path, str(self.cuSize)+'_only_win/module-%d.pkl' % (i))))
            #my_load_state_dict(module,torch.load("../ablation-network/trained_models/10-17-models/"+str(self.cuSize)+'/module-%d.pkl' % (i)))

    def build_model(self):
        self.res=[]
        if self.cuSize<=4:
            self.res.append(single_conv(ch_in=1,ch_out=16))
        else:
            self.res.append(single_conv(ch_in=3,ch_out=16))

        if self.cuSize%5==0:
            self.res.append(Win_noShift_Attention(dim=16, window_size=(8,8),num_heads=4))
            #self.res.append(Win_noShift_Attention(dim=16, window_size=(32,32),num_heads=4))
        elif self.cuSize==1:
            self.res.append(Win_noShift_Attention(dim=16, window_size=(16,16),num_heads=4))
        elif self.cuSize==2:
            self.res.append(Win_noShift_Attention(dim=16, window_size=(4,8),num_heads=4))
        elif self.cuSize==3:
            self.res.append(Win_noShift_Attention(dim=16, window_size=(8,32),num_heads=4))
        elif self.cuSize==4:
            self.res.append(Win_noShift_Attention(dim=16, window_size=(8,16),num_heads=4))
        
        if self.cuSize%5==0:
            self.subnet=subnet2(6)
        elif self.cuSize==1:
            self.subnet=subnet3(6)#mt
        elif self.cuSize==2:
            self.subnet=subnet3(6,x=4,y=8,atten_input=8)
        elif self.cuSize==3:
            self.subnet=subnet4(6,x=4,y=16,atten_input=8)
        elif self.cuSize==4:
            self.subnet=subnet4(6,x=4,y=8,atten_input=12)
            
        self.cuSize_list=[self.res[0],self.res[1],self.subnet]
        
        self.optimizer=optim.Adam(chain(self.res[0].parameters(),self.res[1].parameters(),self.subnet.parameters()) \
                                 , self.lr, [self.beta1, self.beta2])       
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience=4)

        for i in range(2):
            self.res[i]=self.res[i].to(self.device)
        self.subnet=self.subnet.to(self.device)

    def save_model(self):
        if not os.path.exists(os.path.join(self.model_path, str(self.cuSize))):
            os.mkdir(os.path.join(self.model_path, str(self.cuSize)))
        if not os.path.exists(os.path.join(self.model_path, str(self.cuSize))):
            os.mkdir(os.path.join(self.model_path, str(self.cuSize)))
        for i,module in enumerate(self.cuSize_list):
            tmp_model_path = os.path.join(self.model_path, str(self.cuSize)+'/module-%d.pkl' % (i))
            tmp_model = module.state_dict()
            torch.save(tmp_model, tmp_model_path)
    def calculate_loss(self,images,pre,total_acc,total_k2_acc, total_remains,gt_remains,thres=0.1):
        images['gt']=images['gt'].to(self.device)
        sum_loss=0

        soft=torch.nn.functional.softmax(pre[:,0:6],dim=1)
        tmp_sum=torch.sum(torch.log(soft)*images['gt'])
        if torch.isnan(tmp_sum):
            sum_loss-=torch.sum(torch.log(soft+1e-14)*images['gt'])
        else:
            sum_loss-=tmp_sum

        for j in range(images['image'].size(0)):
            if torch.argmax(soft[j])==torch.argmax(images['gt'][j]):
                total_acc+=1
            
            for k in range(6):
                if soft[j][k]>thres:
                    total_k2_acc+=images['gt'][j][k]
                    total_remains+=1

        sum_loss/=images['image'].size(0)
        return total_acc, total_k2_acc, total_remains, sum_loss, gt_remains

    def run(self,images,total_acc,total_k2_acc,total_length,total_remains,gt_remains,thres=0.1):

        pre=self.res[0](images['image'].to(self.device))
        pre=self.res[1](pre)
        pre=self.subnet(pre,images['qp'])

        total_acc, total_k2_acc, total_remains,sum_loss, gt_remains=self.calculate_loss(images,pre,total_acc,total_k2_acc, total_remains,gt_remains,thres)
        
        total_length+=self.batch_size
        return total_acc,total_k2_acc,total_length,total_remains,sum_loss, gt_remains

    def validate(self,valid_loader,thres=0.1):
        with torch.no_grad():
            for module in self.cuSize_list:
                module.train(False)
                module.eval()
            total_acc=0.
            total_length=0
            total_remains=0
            epoch_sum_loss = 0
            total_k2_acc=0.
            gt_remains=0.

            for i, images in enumerate(valid_loader):
                total_acc,total_k2_acc,total_length,total_remains,sum_loss,gt_remains= \
                self.run(images,total_acc,total_k2_acc,total_length,total_remains,gt_remains,thres)
                epoch_sum_loss += sum_loss.item()

            print(
                '[Validation]cuSize:%d, Sum_Loss: %.4f, acc: %.4f, k2_acc: %.4f, remains: %.4f, length: %d, gt_remains: %.4f\n' % \
                    (self.cuSize, epoch_sum_loss/total_length*self.batch_size,total_acc/total_length, total_k2_acc/total_length, total_remains/total_length,total_length, \
                    gt_remains/total_length))
            for module in self.cuSize_list:
                module.train(True)
        self.scheduler.step(epoch_sum_loss)
        print(self.optimizer.param_groups[0]['lr'])
        return total_acc, total_k2_acc, total_length, total_remains

    def train(self):
        train_loader=get_loader(cuSize=self.cuSize, batch_size=self.batch_size, num_workers=8, mode='train')
        valid_loader=get_loader(cuSize=self.cuSize, batch_size=self.batch_size, num_workers=8, mode='valid')
        sum_image=600000
        validation_num=10000 #80000
        total_remains=0
        total_acc=0.
        total_length=0
        epoch_sum_loss=0
        total_k2_acc=0.
        gt_remains=0.
        start = time.time()
        while 1:
            for module in self.cuSize_list:
                module.train(True)

            for i, images in enumerate(train_loader):
                total_acc, total_k2_acc, total_length,total_remains,sum_loss, gt_remains= \
                self.run(images,total_acc,total_k2_acc,total_length,total_remains,gt_remains)
                epoch_sum_loss += sum_loss.item()

                self.optimizer.zero_grad()
                sum_loss.backward()
                self.optimizer.step()
                
                sum_image-=1
                
                if sum_image % validation_num==0:
                    print('Training epoch {} times is {}: '.format(sum_image, time.time() - start))
                    print('[Training]cuSize:%d,Image [%d],Sum_Loss: %.4f, acc: %.4f, k2_acc: %.4f, remains: %.4f,length: %d, gt_remains: %.4f\n' % \
                            (self.cuSize,sum_image, epoch_sum_loss,total_acc/total_length, total_k2_acc/total_length, total_remains/total_length,total_length, \
                            gt_remains/total_length))
                
                    #validation
                    self.validate(valid_loader)
                    if not self.isDebug:
                        self.save_model()
                    total_remains=0
                    total_acc=0.
                    total_length=0
                    epoch_sum_loss=0
                    total_k2_acc=0.
                    gt_remains=0.
                    start = time.time()
                if sum_image<0:
                    break

            if sum_image<0:
                break    
            print("epoch_end")
        torch.cuda.empty_cache()
        gc.collect()