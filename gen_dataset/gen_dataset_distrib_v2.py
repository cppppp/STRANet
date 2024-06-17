import glob, struct, json, os, math
import numpy as np
from cal_prob import cal_prob_v2
cuSize=2
count_splits=[0,0,0,0,0,0]
counti=[0,0,0,0,0,0,0]
all_counti=0
mode0_counti=0
same_counti=0

def set_cus_rdcost(file_name, w, h, setting, cuSize):
    global same_counti
    global mode0_counti
    record_size = struct.calcsize("=2H4B6d1?")
    with open(file_name,'rb') as fd:
        buffer = fd.read()
    
    n = 0
    valid_flag=1

    if len(buffer)%record_size!=0:
        valid_flag=0
    else:
        while n * record_size < len(buffer):
            tmpcost=[0,0,0,0,0,0]
            if setting<=3:
                cux, cuy, cuh, cuw, split, last_split, tmpcost[0], tmpcost[1], tmpcost[2], tmpcost[3], tmpcost[4], tmpcost[5], channel = \
                struct.unpack("=2H4B6d1?", buffer[n * record_size: (n + 1) * record_size])
            else:
                cux, cuy, cuh, cuw, split, last_split, tmpcost[0], tmpcost[1], tmpcost[3], tmpcost[2], tmpcost[5], tmpcost[4], channel = \
                struct.unpack("=2H4B6d1?", buffer[n * record_size: (n + 1) * record_size])
                if split==2 or split==3:
                    split=5-split
                elif split==4 or split==5:
                    split=9-split
            n += 1
            if cuSize<=4 and channel!=False:
                continue
            if cuSize==5 and channel==False:
                continue
            if (cuSize==5 and cuh==64 and cuw==64) or (cuSize==0 and cuh==32 and cuw==32) or (cuSize==1 and cuh==16 and cuw==16) or (cuSize==2 and cuh+cuw==48) or (cuSize==3 and ((cuh==8 and cuw==32) or (cuh==32 and cuw==8))) \
               or  (cuSize==4 and ((cuh==8 and cuw==16) or (cuh==16 and cuw==8))):
                if cuSize==0 or cuSize==5:
                    mode=0
                elif cuSize==1:
                    mode=1
                    if last_split==1:
                        mode=0
                    elif tmpcost[2]==0:
                        mode=2
                    elif tmpcost[3]==0:
                        mode=3
                elif cuSize==2:
                    if setting<=3:
                        if cuh==16 and tmpcost[2]==0:
                            mode=0
                        elif cuh==16:
                            mode=1
                        elif cuw==16 and tmpcost[3]==0:
                            mode=2
                        else:
                            mode=3
                    else:
                        if cuw==16 and tmpcost[2]==0:
                            mode=0
                        elif cuw==16:
                            mode=1
                        elif cuh==16 and tmpcost[3]==0:
                            mode=2
                        else:
                            mode=3
                elif cuSize==3:
                    if setting<=3:
                        if cuh==8 and tmpcost[2]==0:
                            mode=0
                        elif cuh==8:
                            mode=1
                        elif cuw==8 and tmpcost[3]==0:
                            mode=2
                        else:
                            mode=3
                    else:
                        if cuw==8 and tmpcost[2]==0:
                            mode=0
                        elif cuw==8:
                            mode=1
                        elif cuh==8 and tmpcost[3]==0:
                            mode=2
                        else:
                            mode=3
                elif cuSize==4:
                    if setting<=3:
                        if cuh==8 and tmpcost[2]==0:
                            mode=0
                        elif cuh==8 and tmpcost[3]==0:
                            mode=1
                        elif cuh==8:
                            mode=2
                        elif cuw==8 and tmpcost[3]==0:
                            mode=3
                        elif cuw==8 and tmpcost[2]==0:
                            mode=4
                        else:
                            mode=5
                    else:
                        if cuw==8 and tmpcost[2]==0:
                            mode=0
                        if cuw==8 and tmpcost[3]==0:
                            mode=1
                        elif cuw==8:
                            mode=2
                        elif cuh==8 and tmpcost[3]==0:
                            mode=3
                        elif cuh==8 and tmpcost[2]==0:
                            mode=4
                        else:
                            mode=5
                
                if setting==0:
                    new_h=cuy
                    new_w=cux
                elif setting==1:
                    new_h=h-cuy-cuh
                    new_w=cux
                elif setting==2:
                    new_h=cuy
                    new_w=w-cux-cuw
                elif setting==3:
                    new_h=h-cuy-cuh
                    new_w=w-cux-cuw
                elif setting==4:
                    new_h=cux
                    new_w=cuy
                elif setting==5:
                    new_h=cux
                    new_w=w-cuy-cuh
                elif setting==6:
                    new_h=h-cux-cuw
                    new_w=cuy
                elif setting==7:
                    new_h=h-cux-cuw
                    new_w=w-cuy-cuh

                if new_h<0 or new_w<0: #error
                    print(setting)
                
                if str(new_h)+"_"+str(new_w) not in cus[mode].keys():
                    cus[mode][str(new_h)+"_"+str(new_w)]=[split]
                    rdcost[mode][str(new_h)+"_"+str(new_w)]=[[tmpcost[c] for c in range(6)]]
                else:
                    cus[mode][str(new_h)+"_"+str(new_w)].append(split)
                    rdcost[mode][str(new_h)+"_"+str(new_w)].append([tmpcost[c] for c in range(6)])
                    
    return valid_flag


show_distrib=np.zeros((81))

real_partition_error=0
real_partition_num=0


for qp in [37, 32, 27, 22]:
    if not os.path.exists('./real_prob_collected_'+str(cuSize)):
        os.mkdir('./real_prob_collected_'+str(cuSize))
    if not os.path.exists('./real_prob_collected_'+str(cuSize)+'/'+str(qp)):
        os.mkdir('./real_prob_collected_'+str(cuSize)+'/'+str(qp))
    for file in glob.glob("../saved_from_server/run-10.23/compressed_"+str(qp)+"/0/*"):
        valid_flag=1
        cus=[{}, {}, {}, {}, {}, {}] #first: cantQT, second: canQT
        rdcost=[{}, {}, {}, {}, {}, {}]
        prob=[{},{},{},{}, {}, {}]

        with open(file, 'rb') as fd:
            buffer = fd.read()
        n = 0
        w, h = map(int, file.split('_')[-2].split('x'))

        valid_flag*=set_cus_rdcost(file, w, h, 0, cuSize)

        for k in range(1,8):
            valid_flag*=set_cus_rdcost("../saved_from_server/run-10.23/compressed_"+str(qp)+"/"+str(k)+"/"+file.split("/")[-1], w, h, k, cuSize)
        
        if valid_flag==1:
            output_dict={}
            output_dict['cus']=cus

            for mode in range(6):
                for res_key,i in rdcost[mode].items():
                    normed=[]
                    for y in range(len(i)):
                        normed.append([])
                    tmp_distrib=[]
                    
                    # deprecate samples with less than eight values
                    if len(i)<8:
                        continue

                    for idx,j in enumerate(i): # enumerate each rotation type
                        if cuSize%5==0 or ( cuSize==1 and mode==0):
                            avg=sum(j) / 6
                        elif ( cuSize==1 and mode==1) or ( cuSize==2 and mode==1) or ( cuSize==2 and mode==3):
                            avg=sum(j) / 5
                        elif ( cuSize==1 and (mode==2 or mode==3)) or (cuSize==2 and (mode==0 or mode==2)) or (cuSize==3 and (mode==1 or mode==3)) \
                              or (cuSize==4 and (mode==2 or mode==5)):
                            avg=sum(j) / 4
                        elif (cuSize==3 and (mode==0 or mode==2)) or (cuSize==4 and (mode==0 or mode==1 or mode==3 or mode==4)):
                            avg=sum(j) / 3
                        else:
                            print("error")

                        for c in j:
                            if c!=0:
                                normed[idx].append(c-avg)
                            else:
                                normed[idx].append(1000000000000000)

                    for j in range(6):
                        #for modes not tried, set rdcost to a large number, so that its probability will be zero
                        if normed[0][j]==1000000000000000:
                            tmp_distrib.append(1000000000000000)
                            tmp_distrib.append(2000)
                            continue

                        avg,var=0,0

                        for t in range(len(i)):
                            avg+=normed[t][j]

                        avg/=len(i)
                        for t in range(len(i)):
                            normed[t][j]-=avg
                            var+=(normed[t][j])**2
                        var=math.sqrt(var/(len(i)-1))

                        tmp_distrib.append(avg)
                        tmp_distrib.append(var)
                    
                    prob[mode][res_key] = cal_prob_v2(tmp_distrib)

            output_dict['prob']=prob
            with open('./real_prob_collected_'+str(cuSize)+'/'+str(qp)+'/'+file.split("/")[-1].split(".")[0]+".json","w") as write_file:
                json.dump(output_dict, write_file)
        else:
            print(file)