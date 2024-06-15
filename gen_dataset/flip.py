# this code generates the eight rotation types
import glob
import numpy as np
def import_yuv(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=False):
    """Load Y, U, and V channels separately from a 8bit yuv420p video.
    
    Args:
        seq_path (str): .yuv (imgs) path.
        h (int): Height.
        w (int): Width.
        tot_frm (int): Total frames to be imported.
        yuv_type: 420p or 444p
        start_frm (int): The first frame to be imported. Default 0.
        only_y (bool): Only import Y channels.

    Return:
        y_seq, u_seq, v_seq (3 channels in 3 ndarrays): Y channels, U channels, 
        V channels.
    """
    # setup params
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq[0], u_seq[0], v_seq[0]
for video in glob.glob("../../QTMTdataset/raw/*"):
    setting=0 #choose from 0 to 3
    is_rotate=True #choose True or False
    y,u,v=import_yuv(video,int(video.split("_")[-2].split("x")[1]), \
                           int(video.split("_")[-2].split("x")[0]),1)
    if is_rotate:
        unrotated=[y,u,v]
        ori_yuv=[np.zeros((y.shape[1],y.shape[0])),np.zeros((u.shape[1],u.shape[0])),np.zeros((v.shape[1],v.shape[0]))]
        for i in range(3):
            for idx,row in enumerate(unrotated[i]): 
                ori_yuv[i][:,idx]=row
    else:
        ori_yuv=[y,u,v]
    new_yuv=[ori_yuv[0].copy(),ori_yuv[1].copy(),ori_yuv[2].copy()]
    new2_yuv=[ori_yuv[0].copy(),ori_yuv[1].copy(),ori_yuv[2].copy()]
    for i in range(3):
        if setting==1:
            for idx,row in enumerate(ori_yuv[i]): 
                new_yuv[i][-idx-1]=row
        elif setting==2:
            for col in range(ori_yuv[i].shape[1]):
                new_yuv[i][:,ori_yuv[i].shape[1]-1-col]=ori_yuv[i][:,col]
        elif setting==3:
            for idx,row in enumerate(ori_yuv[i]): 
                new2_yuv[i][-idx-1]=row
            for col in range(ori_yuv[i].shape[1]):
                new_yuv[i][:,ori_yuv[i].shape[1]-1-col]=new2_yuv[i][:,col]
    if is_rotate:
        yuv_path="./yuv/"+str(setting+4)+"/"+video.split("/")[-1]
    else:
        yuv_path="./yuv/"+str(setting)+"/"+video.split("/")[-1]
    
    with open(yuv_path, 'wb') as f:
        for channel in new_yuv:
            for elem0 in channel:
                for elem1 in elem0:
                    f.write(np.uint8(elem1)) # 8 bits
