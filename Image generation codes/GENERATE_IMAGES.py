#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:27:39 2021

@author: gaben
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import random2
import math
from Read_Richers import *
from Injection import *
from PIL import Image
from preprocessing import *
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from os import remove
import glob
import pylab as pl

#Create the files path
file_path = 'WRITE THE FILE PATH'
richers_filename = 'WRITE THE RICHERS DATABASE PATH'
out_path = 'WRITE THE OUT PATH'

#Get the list of avaible segments by gps
gps_list = get_segments_gps(file_path)


def create_image(i):   
    #Generate random injection parameters
    #Ramdom gps time for the injection
    nf = random2.randrange(0, len(gps_list))
    gps_ini = int(gps_list[nf])
    gps_end = gps_ini + 4096
    gps0 = random2.randrange(gps_ini,gps_end)
    
    #Parameters
    dur = 4           #Total duration
    t0 = dur/2        #Shift to center the signal (gps should correspond to the signal position) 
    dist = 20     #Distance to the source
    ra = 0     #Sky position
    dec = 0


    #Load noise_files
    files  = get_filename(file_path,gps0)
    bgH = TimeSeries.read(files[0],format='hdf5.losc')
    bgL = TimeSeries.read(files[1],format='hdf5.losc')
    bgV = TimeSeries.read(files[2],format='hdf5.losc')
    
    # Load signal from Richers catalog
    simulation_list = get_Richers_waveforms_name(richers_filename)
    indx = random2.randrange(len(simulation_list))
    gw_name = simulation_list[indx]
    #load signal
    t, h, Deltah, fpeak = read_Richers_file(gw_name,richers_filename)

    
    injection_parameters = dict(Deltah=100, fpeak = 10, dist = dist, theta = np.pi/2,
                                ra = ra, dec = dec, psi = 0, t0 = t0, geocent_time = gps0 -t0,duration=dur,fs=16384,
                            Rich_time=t,Rich_strain=h, noise_type='no-noise',gps0=gps0-t0)
    #Make injection
    ifos = make_bilby_injection(injection_parameters,['H1', 'L1', 'V1'])
    #Convert into TimeSeries
    h1 = ifo_to_TS(ifos[0])
    l1 = ifo_to_TS(ifos[1])
    v1 = ifo_to_TS(ifos[2])
    
    #Make injection
    sigH1,bgH1,h1 = real_noise_injection(bgH,h1,files[0])
    sigL1,bgL1,l1 = real_noise_injection(bgL,l1,files[1])
    sigV1,bgV1,v1 = real_noise_injection(bgV,v1,files[2])
    
    #ASD calculation
    
    gpsS = int(files[0].split('-')[2])
    asdH = asd_calculation(bgH,gpsS,gpsS+4096,file_path,'H')
    asdL = asd_calculation(bgL,gpsS,gpsS+4096,file_path,'L')
    asdV = asd_calculation(bgV,gpsS,gpsS+4096,file_path,'V')
    
    #Whitening

    bg_h1 = bgH1.whiten(asd=asdH).bandpass(20,1000).notch(60).notch(120).notch(240)
    bg_l1 = bgL1.whiten(asd=asdL).bandpass(20,1000).notch(60).notch(120).notch(240)
    bg_v1 = bgV1.whiten(asd=asdV).bandpass(20,1000).notch(50).notch(100).notch(200)

    sig_h1 = sigH1.whiten(asd = asdH).bandpass(20,1000).notch(60).notch(120).notch(200)
    sig_l1 = sigL1.whiten(asd = asdL).bandpass(20,1000).notch(60).notch(120).notch(240)
    sig_v1 = sigV1.whiten(asd = asdV).bandpass(20,1000).notch(50).notch(100).notch(200)
    #Clean variables to reduce memory
    del bgH1 
    del bgL1
    del bgV1
    
    #Ramdom poisition of the signal inside the spectrogram
    tm = random2.uniform(-0.1*3/4, 0.1*3/4)
    
    #Compute Q_transform
    tres=(0.2/256) #To fix size to 256 x 256 image
    fres= 980/256
    frange=(20,1000)
    outseg = (gps0-0.1+tm,gps0+0.1+tm)
    sig_h1_qt=sig_h1.q_transform(outseg = outseg, norm='median', frange=frange,tres=tres,fres= fres,whiten=False)
    sig_l1_qt=sig_l1.q_transform(outseg = outseg, norm='median', frange=frange,tres=tres,fres= fres,whiten=False)
    sig_v1_qt=sig_v1.q_transform(outseg = outseg, norm='median', frange=frange,tres=tres,fres= fres,whiten=False)
    
    bg_h1_qt = bg_h1.q_transform(outseg = outseg, norm='median', frange=frange,tres=tres,fres= fres,whiten=False)
    bg_l1_qt = bg_l1.q_transform(outseg = outseg, norm='median', frange=frange,tres=tres,fres= fres,whiten=False)
    bg_v1_qt = bg_v1.q_transform(outseg = outseg, norm='median', frange=frange,tres=tres,fres= fres,whiten=False)
    
    #Generate images
    sig_h1_qti = sig_h1_qt.value
    sig_l1_qti  = sig_l1_qt.value
    sig_v1_qti  = sig_v1_qt.value
    
    bg_h1_qti = bg_h1_qt.value
    bg_l1_qti = bg_l1_qt.value
    bg_v1_qti = bg_v1_qt.value
    
    #Offset so smallest value i
    sig_h1_qti = Q_to_image(sig_h1_qti) #m=-1 so all the colors are normalized (see Q_to_image in processing)
    bg_h1_qti  = Q_to_image(bg_h1_qti)
    sig_l1_qti = Q_to_image(sig_l1_qti)
    bg_l1_qti  = Q_to_image(bg_l1_qti)
    sig_v1_qti = Q_to_image(sig_v1_qti)
    bg_v1_qti  = Q_to_image(bg_v1_qti)
    
    #Save to files
    sig_rgb_qt=np.array([sig_h1_qti, sig_l1_qti, sig_v1_qti])
    bg_rgb_qt=np.array([bg_h1_qti, bg_l1_qti, bg_v1_qti])
    
     #Signal-tnoise ratio
    snr_h1 = ifos[0].meta_data['optimal_SNR']
    snr_l1 = ifos[1].meta_data['optimal_SNR']
    snr_v1 = ifos[2].meta_data['optimal_SNR']
    
    snr_med = round((snr_h1**2+snr_l1**2+snr_v1**2)**0.5, 2)
    
    #Create and save images (make sure the folders exist)
    sig_im=Image.fromarray(sig_rgb_qt.T)
    sig_im=sig_im.transpose(Image.FLIP_TOP_BOTTOM)
    sig_im.save(f'{out_path}/sig/{i}_{fpeak:.2f}_{Deltah:.4f}_{snr_med}.png')
    
    bg_im=Image.fromarray(bg_rgb_qt.T)
    bg_im=bg_im.transpose(Image.FLIP_TOP_BOTTOM)
    bg_im.save(f'{out_path}/bg/{i}N.png')
       
    return snr_med
  

#Modify num_images to print as images as wanted
num_images = 10000
snr_med, snr_num = [], []
for i in range(num_images):
    snr_med.append(create_image(i))

#Write SNR vectors in .txt documents
with open(f'{out_path}/snr_signals.txt', 'w') as f:
    f.write('SNR medio de los detectores\n [')
    for p in range(12):
        snr_num.append(0)
    for t in snr_med:
        if 0 < t <= 8:
            snr_num [0] = snr_num[0] + 1
        if 8 < t <= 16:
            snr_num [1] = snr_num[1] + 1
        if 16 < t <= 24:
            snr_num [2] = snr_num[2] + 1
        if 24 < t <= 32:
            snr_num [3] = snr_num[3] + 1
        if 32 < t <= 40:
            snr_num [4] = snr_num[4] + 1
        if 40 < t <= 48:
            snr_num [5] = snr_num[5] + 1
        if 48 < t <= 56:
            snr_num [6] = snr_num[6] + 1
        if 56 < t <= 64:
            snr_num [7] = snr_num[7] + 1
        if 64 < t <= 72:
            snr_num [8] = snr_num[8] + 1
        if 72 < t <= 80:
            snr_num [9] = snr_num[9] + 1
        if 80 < t <= 88:
            snr_num [10] = snr_num[10] + 1
        if 88 < t:
            snr_num [10] = snr_num[10] + 1
        
        x = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92]
        pl.rc('lines', linewidth=1, linestyle='--', marker='s', markersize=11,color = 'b')
        pl.rc('xtick', labelsize=20) 
        pl.rc('ytick', labelsize=20) 
        pl.figure(figsize=(20,10))
        pl.plot(x, snr_num, linewidth = 3.5)
        pl.grid()
        pl.ylabel("SNR", fontsize = 25)
        pl.xlabel("Número", fontsize = 25)

        pl.show()
        
        
        if t != snr_med[num_images-1]:
            f.write(str(t) +', ')
        else:
            f.write(str(t) +']')



    

