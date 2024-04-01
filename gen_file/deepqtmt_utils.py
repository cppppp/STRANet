import torch
import numpy as np
import cv2

# ==========
# YUV420P
# ==========

__all__ = ['import_yuv']

def import_yuv_10bit(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=False):
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w * 2, hh * ww * 2, hh * ww * 2
    blk_size = y_size + u_size + v_size
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.float)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.float)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.float)
    
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)) * 8, 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint16, count=y_size//2).reshape(h, w)/4
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint16, \
                    count=u_size//2).reshape(hh, ww)/4
                v_frm = np.fromfile(fp, dtype=np.uint16, \
                    count=v_size//2).reshape(hh, ww)/4
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm
    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq

def import_yuv_4frame(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=False):
    if "MarketPlace" in seq_path or \
        "RitualDance" in seq_path or \
        "DaylightRoad2" in seq_path or \
        "FoodMarket4" in seq_path or \
        "ParkRunning3" in seq_path or \
        "Tango2" in seq_path or \
        "Campfire" in seq_path or \
        "CatRobot" in seq_path:
        return import_yuv_10bit(seq_path, h, w, tot_frm, yuv_type, start_frm, only_y)
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
            fp.seek(int(blk_size * (start_frm + i) * 8), 0)  # skip frames
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
        return y_seq, u_seq, v_seq

def copy_value(a,b):
    c=a
    d=b
    return c,d