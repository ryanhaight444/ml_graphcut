# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
import torch.nn as nn
from scipy.io import loadmat
import skvideo.io
import skvideo.datasets
# Our libs
from dataset import VidDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm

import time
import dim_reduction
import operator
import pandas
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolor
from matplotlib.patches import Patch
import matplotlib.image as mpimg


colors = loadmat('data/color150.mat')['colors']

classes = pandas.read_csv('data/object150_info.csv')['Name'].values
for i,name in enumerate(classes):
    name = name.split(';')[0]
    classes[i] = name 

class_colors = {}

for i,l in enumerate(classes):
    clr = colors[i]
    clr = (clr[0]/255, clr[1]/255, clr[2]/255)
    class_colors[l] = clr

def create_gif(args):
    home = os.getcwd()
    os.chdir(args.out_dir)
    frame_dir = args.out_dir
    gif_array = []
    writer = skvideo.io.FFmpegWriter("outputvideo.mp4", outputdict={'-b': '30000000'})
    for frame in sorted(os.listdir(os.getcwd())):
        frame_img = cv2.imread(frame)
        frame_array = np.asarray(frame_img)
        #BGR -> RGB
        frame_array = frame_array[:,:,::-1]
        for i in range(0,4):        
            writer.writeFrame(frame_array)
    
    writer.close()

    os.chdir(home)

def visualize_result(labels,name,frame):
    label_counts = {}
    patches = []
    legend_classes = []
    for row in labels:
        for label in row:
            label = classes[label]
            
            try:
                label_counts[label] = label_counts[label] + 1
            except KeyError as e:
                label_counts[label] = 1
    

    label_counts_sorted = sorted(label_counts.items(), key=operator.itemgetter(1))
    label_counts_sorted.reverse()
    for l in label_counts_sorted:
        legend_classes.append(l[0])
        patches.append(Patch(color = class_colors[l[0]], label = 'Color Patch'))

    # prediction
    pred_color = colorEncode(labels, colors)
    #BGR -> RGB
    pred_color = pred_color[:,:,::-1]

################
    fig, ax = plt.subplots(1,2,figsize=(20, 9))
    plt.sca(ax[0])                # set the current axes instance to the top left
    ax[0].imshow(frame)
    plt.sca(ax[1])                # set the current axes instance 
    ax[1].imshow(pred_color)

    l = ax[1].legend(patches, legend_classes,
             title="Labels",
             loc="center left",
             bbox_to_anchor=(1, 0.2, 0.5, 1),
             ncol=(len(patches)//30) + 1)



#################
    print('saved figure for ' + name)
    plt.savefig(name + '.png')
    plt.close()

parser = argparse.ArgumentParser()

parser.add_argument('--label_dir', required=True, type=str,
                        help='a path to a director of labels for frames to create visualization')
parser.add_argument('--out_dir', required=True, type=str,
                        help='a path to a video to generate visualization')
parser.add_argument ('--video', required = True, type=str,
                        help='a path to the video')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

labels_list = sorted(os.listdir(args.label_dir))
video_array = skvideo.io.vread(args.video)

for i,frame_file in enumerate(labels_list):
    file_name = str(args.label_dir)+frame_file
    labels = np.load(file_name)
    home = os.getcwd()
    os.chdir(args.out_dir)
    frame_image = video_array[i]
    visualize_result(labels,frame_file,frame_image)
    os.chdir(home)
create_gif(args)



