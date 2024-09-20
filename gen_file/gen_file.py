from network import *
from STRANet_utils import import_yuv_4frame,copy_value
import os,torch,time
from new_stf import Win_noShift_Attention
import gc
import json
from statistics import mean
import argparse
device = torch.device('cuda:0')
batch_size_0=512
batch_size_1=512
batch_size_2=512
batch_size_3=512
batch_size_4=512
batch_size_5=512

speed_choice="C2/"
thres=0.15

def main(file_name, model_path, is_Window, tot_frm):
    if not os.path.exists("./"+speed_choice):
        os.mkdir("./"+speed_choice)
    if not os.path.exists("./"+speed_choice+file_name.split("/")[-1]):
        os.mkdir("./"+speed_choice+file_name.split("/")[-1])
    img_h=int(file_name.split("_")[-2].split("x")[1])
    img_w=int(file_name.split("_")[-2].split("x")[0])
    
    model_time=[]
        
    for qp in range(4):
        start_time=time.time()
        #build model
        res=[]
        if is_Window:
            res.append(single_conv(ch_in=1,ch_out=16))
            res.append(Win_noShift_Attention(dim=16, window_size=(8,8),num_heads=4))
            res.append(single_conv(ch_in=1,ch_out=16))
            res.append(Win_noShift_Attention(dim=16, window_size=(16,16),num_heads=4))
            res.append(single_conv(ch_in=1,ch_out=16))
            res.append(Win_noShift_Attention(dim=16, window_size=(4,8),num_heads=4))
            res.append(single_conv(ch_in=1,ch_out=16))
            res.append(Win_noShift_Attention(dim=16, window_size=(8,32),num_heads=4))
            res.append(single_conv(ch_in=1,ch_out=16))
            res.append(Win_noShift_Attention(dim=16, window_size=(8,16),num_heads=4))
            res.append(single_conv(ch_in=3,ch_out=16))
            res.append(Win_noShift_Attention(dim=16, window_size=(8,8),num_heads=4))
        else:
            for i in range(5):
                res.append(res12(1))#mt
                res.append(res3(16))
            res.append(res12(3))#mt
            res.append(res3(16))
        
        subnet=[]
        subnet.append(subnet2(6))
        subnet.append(subnet3(6))
        subnet.append(subnet3(6,x=4,y=8,atten_input=8))
        subnet.append(subnet4(6,x=4,y=16,atten_input=8))
        subnet.append(subnet4(6,x=4,y=8,atten_input=12))
        subnet.append(subnet2(6))
        
        cuSize_list=[
            [res[0],res[1],subnet[0]],  #32x32
            [res[2],res[3],subnet[1]],  #16x16
            [res[4],res[5],subnet[2]],
            [res[6],res[7],subnet[3]],
            [res[8],res[9],subnet[4]],
            [res[10],res[11],subnet[5]]
        ]

        for cuSize in range(len(cuSize_list)):
            for i,module in enumerate(cuSize_list[cuSize]):
                module.load_state_dict(torch.load(os.path.join(model_path+str(cuSize)+'/module-%d.pkl' % (i))), strict=True)
                module.to(device)
                module.train(False)
                module.eval()

        cus={} #key:frame,cux,cuy,cuh,cuw value:predicted partitions

        max_pool=torch.nn.MaxPool2d((2,2))
        y, u, v = import_yuv_4frame(file_name, img_h, img_w, tot_frm, yuv_type='420p', start_frm=0, only_y=False)
        y=torch.unsqueeze(torch.tensor(y),1).float()
        y_down=max_pool(y)
        u=torch.unsqueeze(torch.tensor(u),1)
        v=torch.unsqueeze(torch.tensor(v),1)
        yuv_list=[y_down,u,v]
        yuv_input=torch.unsqueeze(torch.cat(yuv_list,dim=1),1)

        y=torch.unsqueeze(y,1)

        input_list=[]
        pos_list=[]
        for frame in range(tot_frm):
            for h in range(yuv_input.shape[3]//32):
                for w in range(yuv_input.shape[4]//32):
                    input_list.append(yuv_input[frame,:,:,h*32:h*32+32,w*32:w*32+32])
                    pos_list.append((frame,h*64,w*64))
        input_batch=torch.cat(input_list,0).float()
        for k in range(input_batch.shape[0]//batch_size_5+1):
            if input_batch.shape[0]==k*batch_size_5:
                continue
            end_idx=min(input_batch.shape[0],(k+1)*batch_size_5)

            pre=res[10](input_batch[k*batch_size_5:end_idx].to(device))
            pre=res[11](pre)
            pre=subnet[5](pre,torch.ones(pre.shape[0],dtype=int)*qp)
            pre=torch.nn.functional.softmax(pre[:,:6],dim=1).cpu()

            for idx,item in enumerate(pre):
                frame=int(pos_list[idx+k*batch_size_5][0])
                posh=int(pos_list[idx+k*batch_size_5][1])
                posw=int(pos_list[idx+k*batch_size_5][2])

                split_list=[1 if s>thres else 0 for s in item] 

                if sum(split_list)==0:
                    split_list[torch.argmax(item)]=1
                cus[(frame,posh,posw,0,0,0)]=split_list
        
        torch.cuda.empty_cache()

        input_list=[]
        pos_list=[]
        
        # crop to 32x32
        for frame in range(tot_frm):
            for h in range(y.shape[3]//32):
                for w in range(y.shape[4]//32):
                    input_list.append(y[frame,:,:,h*32:h*32+32,w*32:w*32+32])
                    pos_list.append((frame,h*32,w*32))
        input_batch=torch.cat(input_list,0).float()
        
        input_list_2, input_list_3, input_list_4, input_list_5 = [], [], [], []
        pos_list_2, pos_list_3, pos_list_4, pos_list_5 = [], [], [], []
        qp_list_2, qp_list_3, qp_list_4, qp_list_5=[], [], [], []
        
        for k in range(input_batch.shape[0]//batch_size_0+1):
            if input_batch.shape[0]==k*batch_size_0:
                continue
            end_idx=min(input_batch.shape[0],(k+1)*batch_size_0)
            
            pre=res[0](input_batch[k*batch_size_0:end_idx].to(device))
            pre=res[1](pre)
            pre=subnet[0](pre,torch.ones(pre.shape[0],dtype=int)*qp)
            pre=torch.nn.functional.softmax(pre[:,:6],dim=1).cpu()
        
            for idx,item in enumerate(pre):
                frame=int(pos_list[idx+k*batch_size_0][0])
                posh=int(pos_list[idx+k*batch_size_0][1])
                posw=int(pos_list[idx+k*batch_size_0][2])

                split_list=[1 if s>thres else 0 for s in item] 
                if sum(split_list)==0:
                    split_list[torch.argmax(item)]=1

                cus[(frame,posh,posw,32,32,0)]=split_list

                if split_list[1]==1:
                    input_list_3.append(y[frame,:,:,posh:posh+16,posw:posw+16])
                    pos_list_3.append((frame,posh,posw))
                    input_list_3.append(y[frame,:,:,posh+16:posh+32,posw:posw+16])
                    pos_list_3.append((frame,posh+16,posw))
                    input_list_3.append(y[frame,:,:,posh:posh+16,posw+16:posw+32])
                    pos_list_3.append((frame,posh,posw+16))
                    input_list_3.append(y[frame,:,:,posh+16:posh+32,posw+16:posw+32])
                    pos_list_3.append((frame,posh+16,posw+16))
                    qp_list_3+=[0,0,0,0]
                if split_list[2]==1:
                    input_list_2.append(y[frame,:,:,posh:posh+16,posw:posw+32])
                    pos_list_2.append((frame,posh,posw,16,32))
                    input_list_2.append(y[frame,:,:,posh+16:posh+32,posw:posw+32])
                    pos_list_2.append((frame,posh+16,posw,16,32))
                    qp_list_2+=[1,1]
                if split_list[3]==1:
                    input_list_2.append(y[frame,:,:,posh:posh+32,posw:posw+16].transpose(3,2))
                    pos_list_2.append((frame,posh,posw,32,16))
                    input_list_2.append(y[frame,:,:,posh:posh+32,posw+16:posw+32].transpose(3,2))
                    pos_list_2.append((frame,posh,posw+16,32,16))
                    qp_list_2+=[1,1]
                if split_list[4]==1:
                    input_list_2.append(y[frame,:,:,posh+8:posh+24,posw:posw+32])
                    pos_list_2.append((frame,posh+8,posw,16,32))
                    qp_list_2.append(0)
                    input_list_4.append(y[frame,:,:,posh:posh+8,posw:posw+32])
                    pos_list_4.append((frame,posh,posw,8,32,1))
                    input_list_4.append(y[frame,:,:,posh+24:posh+32,posw:posw+32])
                    pos_list_4.append((frame,posh+24,posw,8,32,1))
                    qp_list_4+=[1,1]

                if split_list[5]==1:
                    input_list_2.append(y[frame,:,:,posh:posh+32,posw+8:posw+24].transpose(3,2))
                    pos_list_2.append((frame,posh,posw+8,32,16))
                    qp_list_2.append(0)
                    input_list_4.append(y[frame,:,:,posh:posh+32,posw:posw+8].transpose(3,2))
                    pos_list_4.append((frame,posh,posw,32,8,1))
                    input_list_4.append(y[frame,:,:,posh:posh+32,posw+24:posw+32].transpose(3,2))
                    pos_list_4.append((frame,posh,posw+24,32,8,1))
                    qp_list_4+=[1,1]
        torch.cuda.empty_cache()

        #16x32模型一次运行
        input_batch=torch.cat(input_list_2,0).float()
        for k in range(input_batch.shape[0]//batch_size_1+1):
            if input_batch.shape[0]==k*batch_size_1:
                continue
            end_idx=min(input_batch.shape[0],(k+1)*batch_size_1)
            
            pre=res[4](input_batch[k*batch_size_1:end_idx].to(device))
            pre=res[5](pre)
            pre=subnet[2](pre,torch.tensor(qp_list_2[k*batch_size_1:end_idx])+torch.ones(pre.shape[0],dtype=int)*qp*2)
            pre=torch.nn.functional.softmax(pre[:,:6],dim=1).cpu()
            
            for idx,item in enumerate(pre):
                frame=int(pos_list_2[idx+k*batch_size_1][0])
                posh=int(pos_list_2[idx+k*batch_size_1][1])
                posw=int(pos_list_2[idx+k*batch_size_1][2])
                cuh=int(pos_list_2[idx+k*batch_size_1][3])
                cuw=int(pos_list_2[idx+k*batch_size_1][4])

                split_list=[1 if s>thres else 0 for s in item] 
                if sum(split_list)==0:
                    split_list[torch.argmax(item)]=1
                
                if split_list[2]==1: #16x32->8x32
                    if cuh==16:
                        input_list_4.append(y[frame,:,:,posh:posh+8,posw:posw+32])
                        pos_list_4.append((frame,posh,posw,8,32,2))
                        input_list_4.append(y[frame,:,:,posh+8:posh+16,posw:posw+32])
                        pos_list_4.append((frame,posh+8,posw,8,32,2))
                        qp_list_4+=[1,1]
                    else:
                        input_list_4.append(y[frame,:,:,posh:posh+32,posw:posw+8].transpose(3,2))
                        pos_list_4.append((frame,posh,posw,32,8,2))
                        input_list_4.append(y[frame,:,:,posh:posh+32,posw+8:posw+16].transpose(3,2))
                        pos_list_4.append((frame,posh,posw+8,32,8,2))
                        qp_list_4+=[1,1]
                if split_list[3]==1: #16x32-> 16x16
                    input_list_3.append(y[frame,:,:,posh:posh+16,posw:posw+16])
                    pos_list_3.append((frame,posh,posw,16,16))
                    qp_list_3.append(1)
                    if cuh==16:
                        input_list_3.append(y[frame,:,:,posh:posh+16,posw+16:posw+32])
                        pos_list_3.append((frame,posh,posw+16,16,16))
                        qp_list_3.append(1)
                    else:
                        input_list_3.append(y[frame,:,:,posh+16:posh+32,posw:posw+16])
                        pos_list_3.append((frame,posh+16,posw,16,16))
                        qp_list_3.append(1)
                if split_list[4]==1:
                    if cuh==16:
                        input_list_4.append(y[frame,:,:,posh+4:posh+12,posw:posw+32])
                        pos_list_4.append((frame,posh+4,posw,8,32,2))
                        qp_list_4.append(0)
                    else:
                        input_list_4.append(y[frame,:,:,posh:posh+32,posw+4:posw+12].transpose(3,2))
                        pos_list_4.append((frame,posh,posw+4,32,8,2))
                        qp_list_4.append(0)
                if split_list[5]==1:  #16x32-> 16x16, 8x16
                    if cuh==16:
                        input_list_3.append(y[frame,:,:,posh:posh+16,posw+8:posw+24])
                        pos_list_3.append((frame,posh,posw+8))
                        qp_list_3.append(3) #forbid VBT in the next step

                        input_list_5.append(y[frame,:,:,posh:posh+16,posw:posw+8].transpose(3,2))
                        pos_list_5.append((frame,posh,posw,16,8,2))
                        input_list_5.append(y[frame,:,:,posh:posh+16,posw+24:posw+32].transpose(3,2))
                        pos_list_5.append((frame,posh,posw+24,16,8,2))
                        qp_list_5 += [2,2]
                        
                    else:
                        input_list_3.append(y[frame,:,:,posh+8:posh+24,posw:posw+16])
                        pos_list_3.append((frame,posh+8,posw))
                        qp_list_3.append(2)

                        input_list_5.append(y[frame,:,:,posh:posh+8,posw:posw+16])
                        pos_list_5.append((frame,posh,posw,8,16,2))
                        input_list_5.append(y[frame,:,:,posh+24:posh+32,posw:posw+16])
                        pos_list_5.append((frame,posh+24,posw,8,16,2))
                        qp_list_5 += [2,2]
                    
                if cuh==32:
                    split_list[5],split_list[4]=copy_value(split_list[4],split_list[5])
                    split_list[3],split_list[2]=copy_value(split_list[2],split_list[3])
                cus[(frame,posh,posw,cuh,cuw,0)]=split_list
        torch.cuda.empty_cache()

        #16x16 model inference
        input_batch=torch.cat(input_list_3,0).float()
        for k in range(input_batch.shape[0]//batch_size_2+1):
            if input_batch.shape[0]==k*batch_size_2:
                continue
            end_idx=min(input_batch.shape[0],(k+1)*batch_size_2)
            
            pre=res[2](input_batch[k*batch_size_2:end_idx].to(device))
            pre=res[3](pre)
            pre=subnet[1](pre,torch.tensor(qp_list_3[k*batch_size_2:end_idx])+torch.ones(pre.shape[0],dtype=int)*qp*4)
            pre=torch.nn.functional.softmax(pre[:,:6],dim=1).cpu()
            
            for idx,item in enumerate(pre):
                frame=int(pos_list_3[idx+k*batch_size_2][0])
                posh=int(pos_list_3[idx+k*batch_size_2][1])
                posw=int(pos_list_3[idx+k*batch_size_2][2])
                cuh=cuw=16
                split_list=[1 if s>thres else 0 for s in item] 
                if sum(split_list)==0:
                    split_list[torch.argmax(item)]=1
                cus[(frame, posh, posw, cuh, cuw, qp_list_3[k*batch_size_2+idx])]=split_list

                if qp_list_3[idx + k*batch_size_2] !=0:
                    continue #8x16 will not split in the next step

                if split_list[2]==1: #16x16 -> 8x16
                    input_list_5.append(y[frame,:,:,posh:posh+8,posw:posw+16])
                    pos_list_5.append((frame,posh,posw,8,16))
                    input_list_5.append(y[frame,:,:,posh+8:posh+16,posw:posw+16])
                    pos_list_5.append((frame,posh+8,posw,8,16))
                    qp_list_5+=[2,2]
                if split_list[3]==1:
                    input_list_5.append(y[frame,:,:,posh:posh+16,posw:posw+8].transpose(3,2))
                    pos_list_5.append((frame,posh,posw,16,8))
                    input_list_5.append(y[frame,:,:,posh:posh+16,posw+8:posw+16].transpose(3,2))
                    pos_list_5.append((frame,posh,posw+8,16,8))
                    qp_list_5+=[2,2]
                if split_list[4]==1:
                    input_list_5.append(y[frame,:,:,posh+4:posh+12,posw:posw+16])
                    pos_list_5.append((frame,posh+4,posw,8,16))
                    qp_list_5.append(0) #forbid HBT in the next step
                if split_list[5]==1:
                    input_list_5.append(y[frame,:,:,posh:posh+16,posw+4:posw+12].transpose(3,2))
                    pos_list_5.append((frame,posh,posw+4,16,8))
                    qp_list_5.append(0) 
        torch.cuda.empty_cache()

        #8x32 model inference
        input_batch=torch.cat(input_list_4,0).float()
        for k in range(input_batch.shape[0]//batch_size_3+1):
            if input_batch.shape[0]==k*batch_size_3:
                continue
            end_idx=min(input_batch.shape[0],(k+1)*batch_size_3)
            
            pre=res[6](input_batch[k*batch_size_3:end_idx].to(device))
            pre=res[7](pre)
            pre=subnet[3](pre,torch.tensor(qp_list_4[k*batch_size_3:end_idx])+torch.ones(pre.shape[0],dtype=int)*qp*2)
            pre=torch.nn.functional.softmax(pre[:,:6],dim=1).cpu()
            
            for idx,item in enumerate(pre):
                frame=int(pos_list_4[idx+k*batch_size_3][0])
                posh=int(pos_list_4[idx+k*batch_size_3][1])
                posw=int(pos_list_4[idx+k*batch_size_3][2])
                cuh=int(pos_list_4[idx+k*batch_size_3][3])
                cuw=int(pos_list_4[idx+k*batch_size_3][4])

                split_list=[1 if s>thres else 0 for s in item] 
                if sum(split_list)==0:
                    split_list[torch.argmax(item)]=1

                if int(pos_list_4[idx+k*batch_size_3][5])==2:
                    if cuh==32:
                        split_list[5],split_list[4]=copy_value(split_list[4],split_list[5])
                        split_list[3],split_list[2]=copy_value(split_list[2],split_list[3])
                    cus[(frame, posh, posw, cuh, cuw, qp_list_4[k*batch_size_3+idx])]=split_list
                    continue
                
                if split_list[3]==1: #8x32 -> 8x16
                    if cuh==8:
                        input_list_5.append(y[frame,:,:,posh:posh+8,posw:posw+16])
                        pos_list_5.append((frame,posh,posw,8,16))
                        input_list_5.append(y[frame,:,:,posh:posh+8,posw+16:posw+32])
                        pos_list_5.append((frame,posh,posw+16,8,16))
                        qp_list_5+=[2,2]
                    else:
                        input_list_5.append(y[frame,:,:,posh:posh+16,posw:posw+8].transpose(3,2))
                        pos_list_5.append((frame,posh,posw,16,8))
                        input_list_5.append(y[frame,:,:,posh+16:posh+32,posw:posw+8].transpose(3,2))
                        pos_list_5.append((frame,posh+16,posw,16,8))
                        qp_list_5+=[2,2]
                if split_list[5]==1:
                    if cuh==8:
                        input_list_5.append(y[frame,:,:,posh:posh+8,posw+8:posw+24])
                        pos_list_5.append((frame,posh,posw+8,8,16))
                        qp_list_5.append(1) #forbid VBT in the next step
                    else:
                        input_list_5.append(y[frame,:,:,posh+8:posh+24,posw:posw+8].transpose(3,2))
                        pos_list_5.append((frame,posh+8,posw,16,8))
                        qp_list_5.append(1)
                        
                if cuh==32:
                    split_list[5],split_list[4]=copy_value(split_list[4],split_list[5])
                    split_list[3],split_list[2]=copy_value(split_list[2],split_list[3])
                cus[(frame, posh, posw, cuh, cuw, qp_list_4[k*batch_size_3+idx])]=split_list
        torch.cuda.empty_cache()
        
        #8x16 model inference
        input_batch=torch.cat(input_list_5,0).float()
        for k in range(input_batch.shape[0]//batch_size_4+1):
            if input_batch.shape[0]==k*batch_size_4:
                continue
            end_idx=min(input_batch.shape[0],(k+1)*batch_size_4)
            
            pre=res[8](input_batch[k*batch_size_4:end_idx].to(device))
            pre=res[9](pre)
            pre=subnet[4](pre,torch.tensor(qp_list_5[k*batch_size_4:end_idx])+torch.ones(pre.shape[0],dtype=int)*qp*3)
            pre=torch.nn.functional.softmax(pre[:,:6],dim=1).cpu()
            
            for idx,item in enumerate(pre):
                frame=int(pos_list_5[idx+k*batch_size_4][0])
                posh=int(pos_list_5[idx+k*batch_size_4][1])
                posw=int(pos_list_5[idx+k*batch_size_4][2])
                cuh=int(pos_list_5[idx+k*batch_size_4][3])
                cuw=int(pos_list_5[idx+k*batch_size_4][4])
                split_list=[1 if s>thres else 0 for s in item] 

                if sum(split_list)==0:
                    split_list[torch.argmax(item)]=1
                if cuh==16:
                    split_list[5],split_list[4]=copy_value(split_list[4],split_list[5])
                    split_list[3],split_list[2]=copy_value(split_list[2],split_list[3])
                cus[(frame, posh, posw, cuh, cuw, qp_list_5[k*batch_size_4+idx])]=split_list
        torch.cuda.empty_cache()

        #write to txt file
        with open("./"+speed_choice+file_name.split("/")[-1]+"/"+str(qp)+'.txt','w') as file:
            for key,v in cus.items():
                file.write(str(key[0])+" "+str(key[1])+" "+str(key[2])+" "+str(key[3])+" "+str(key[4])+" "+str(key[5]))
                for split in range(6):
                    file.write(" "+str(v[split]))
                file.write("\n")
            file.write(str(-1)+" "+str(-1)+" "+str(-1)+" "+str(-1)+" "+str(-1)+"\n")
        print(qp)
        model_time.append(time.time()-start_time)

    ''' calculate time overhead
    TS_list=[]
    with open("./baseline.json","r") as json_file:
        dic = json.load(json_file)
        for qp_num in range(22,42,5):
            anchor_time=dic['%d-%s-real_time' % (qp_num,file_name)]
            TS_list.append(model_time[(qp_num-22)//5]/anchor_time)
    return mean(TS_list)'''

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./trained_models_Window/")
    parser.add_argument('--seq_path', type=str, default="/S3/hzj/yuv/720p/")
    parser.add_argument('--is_Window', type=int, default=1)
    parser.add_argument('--tot_frm', type=int, default=4)
    config = parser.parse_args()
    video_list=[
        "Johnny_1280x720_60.yuv"
    ]

    '''video_list=[
        "Campfire_3840x2160_30.yuv",  
        "FoodMarket4_3840x2160_60.yuv",
        "Tango2_3840x2160_60.yuv",
        "CatRobot_3840x2160_60.yuv",
        "DaylightRoad2_3840x2160_60.yuv", #5
        "ParkRunning3_3840x2160_50.yuv",
        "MarketPlace_1920x1080_60.yuv",
        "RitualDance_1920x1080_60.yuv",
        "BasketballDrive_1920x1080_50.yuv",
        "BQTerrace_1920x1080_60.yuv",  #10
        "Cactus_1920x1080_50.yuv",
        "BasketballDrill_832x480_50.yuv",
        "BQMall_832x480_60.yuv",
        "PartyScene_832x480_50.yuv", 
        "RaceHorsesC_832x480_30.yuv", #15
        "RaceHorses_416x240_30.yuv",
        "BQSquare_416x240_60.yuv",  
        "BlowingBubbles_416x240_50.yuv",
        "BasketballPass_416x240_50.yuv",
        "FourPeople_1280x720_60.yuv",  #20
        "Johnny_1280x720_60.yuv",
        "KristenAndSara_1280x720_60.yuv"
    ]'''

    time_sum=0
    for video_item in video_list:
        print(video_item,"start")
        start=time.time()
        time_sum+=main(config.seq_path+video_item,config.model_path,config.is_Window,config.tot_frm)
        print(time.time()-start)
    print("average:",time_sum/22)