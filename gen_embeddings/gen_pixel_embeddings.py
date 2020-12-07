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


colors = loadmat('data/color150.mat')['colors']



def test(segmentation_module, loader, args):
    segmentation_module.eval()
    embeddings = []
    pbar = tqdm(total=len(loader))
    i = 0

    if(args.use_last_layer):
        outdim = 150
    else:
        outdim = 4096
    #where embeddings are located
    out_dir = ""
    if(args.out_dir!= None):
        out_dir = args.out_dir
    else:
        out_dir = args.test_video + "_embeddings_"+ outdim +"/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for batch_data in loader:
        total_start = time.time()
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']
        with torch.no_grad():

            scores = torch.zeros(1, outdim, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            start = time.time()
            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, args.gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(args.imgSize)
            end = time.time()
            start = time.time()
            scores_cpu = scores.cpu()
            scores_cpu = scores_cpu.numpy()
            #save embeddings

            embeddings = scores_cpu.transpose(0,2,3,1)[0]
            np.save(out_dir + "frame" + str(i).zfill(6)+ "_embeddings", embeddings)
            end = time.time()
            print(embeddings.shape)
            i+=1
        total_end = time.time()
        pbar.update(1)




def main(args):
    torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=args.use_softmax,
        use_last_layer=args.use_last_layer)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    video_data = skvideo.io.vread(args.test_video)
    print(video_data.shape)
    dataset_test = VidDataset(
        video_data, args, max_sample=len(video_data))
    loader_test = torchdata.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, args)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--test_video', required=True, type=str,
                        help='a path to a video to generate pixel embeddings on')
    parser.add_argument('--out_dir', default = None, type=str,
                        help='a path to output embeddings')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether or not to use softmax when getting embeddings (this will gen class labels)')
    parser.add_argument('--use_last_layer', action='store_true',
                        help='whether or not to use the final layer of the net - only works for ppmdeepsup, true will gen 150 dim logits false will gen 4096 embeddings')
    parser.add_argument('--model_path', default='baseline-resnet50dilated-ppm_deepsup',
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='./',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')

    args = parser.parse_args()
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
