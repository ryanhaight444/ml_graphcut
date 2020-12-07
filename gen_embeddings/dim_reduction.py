import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import random_projection


def gen_random_projection(frame_array, new_size):
    frame_array_reshaped = frame_array.reshape(-1,frame_array.shape[2])
    transformer = random_projection.GaussianRandomProjection(n_components =
    new_size,random_state = 1)

    projected_frame_array = transformer.fit_transform(frame_array_reshaped)
    projected_frame_array = projected_frame_array.reshape(frame_array.shape[0],
    frame_array.shape[1], -1)
    return(projected_frame_array)

#make labels from logits
def make_labels(logits):
    pred = np.argmax(logits,axis=2)
    return pred

def plot_svd(frame):
    frame_reshaped = frame.reshape(-1,frame_array.shape[2])
    frame_reshaped = np.linalg.svd(frame_reshaped, compute_uv=False)


    plt.plot(frame_reshaped)
    plt.show()
    plt.savefig("marine_01_first_frame")
    plt.plot(frame_reshaped[:100])
    plt.show()
    plt.savefig("marine_01_first_frame")


    frame_reshaped = frame_reshaped.reshape(frame.shape[0],
    frame.shape[1], -1)
    print(frame_reshaped.shape)

def create_rand_mask(height, pixel_prob):
    mask_dir = "random_masks/"
    if(not os.path.exists(mask_dir)):
        os.makedirs(mask_dir)
    if(os.path.isfile(mask_dir + str(height)+str(pixel_prob)+"rand_mask.npy")):
        mask = np.load(mask_dir + str(height)+str(pixel_prob)+"rand_mask.npy")
    else:
        mask = np.random.choice([True, False], size=height, p=[pixel_prob, 1-pixel_prob])
        np.save(mask_dir + str(height)+str(pixel_prob)+"rand_mask.npy",mask)
    return mask

def save_frame(save_dir,frame,i):
    start = time.time()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+'frame' + str(i).zfill(6), frame)


def subsample_pixels(frame_dir,frames,frames_array,mask):
    m_index = 0
    f_index = 0
    print(frame_subsample_rate)
    for i,frame in enumerate(frames):
        if( i%frame_subsample_rate == 0):
            frame = np.load(frame_dir+frame)
            for row in frame:
                for pixel in row:
                    if(mask[m_index] == True):
                        frames_array[f_index] = pixel
                        f_index += 1
                    m_index += 1

    return frames_array

def subsample_labels(label_dir,labels,labels_array,mask):
    m_index = 0
    l_index = 0
    for i,label in enumerate(labels):
        if( i%frame_subsample_rate == 0):
            label = np.load(label_dir+label)
            for row in label:
                for pixel in row:
                    if(mask[m_index] == True):
                        labels_array[l_index] = pixel
                        l_index += 1
                    m_index += 1
    #print(frames_array)
    #print (index)
    return labels_array

#t is the function that applies the transformation
def apply_transformation(frame_array, t):

    frame_reshaped = frame_array.reshape(-1,frame_array.shape[2])

    projected_frame = t.transform(frame_reshaped)

    projected_frame = projected_frame.reshape(frame_array.shape[0],
    frame_array.shape[1], -1)
    print(projected_frame.shape)
    return projected_frame



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--frame_dir', required=True, type=str,
                        help='a path to a video to generate pixel embeddings on')
    parser.add_argument('--label_dir', type=str,
                        help='labels for labeled dim redution')
    parser.add_argument('--new_frame_dir', required=False, type=str,
                        help='The directory to save the projected embeddings')
    parser.add_argument('--mode', required=True, type=str,
                        help='What type of dim reduction you want to use')
    parser.add_argument('--new_size', required=False, type=int,
                        help='The size that we want the embeddings to be' +
                        ' projected to')


    args = parser.parse_args()
    frames = sorted(os.listdir(args.frame_dir))
    labels = sorted(os.listdir(args.label_dir))
    i = 0

    projected_frame = None




    if(args.mode == 'PCA'):
        frame_subsample_rate = 1 #grab every 'nth' frame
        pixel_prob = .09 #each pixel has a 'k%' chance of being sampled

        frame_shape = np.load(args.frame_dir+frames[0]).shape

        num_frames = len(frames)//frame_subsample_rate

        height = num_frames * frame_shape[0] * frame_shape[1]
        width = frame_shape[2]



        mask = create_rand_mask(height, pixel_prob)
        #Creating array to fit to- of subsampled frames from videos
        frames_array = np.zeros((np.sum(mask),width))
        frames_array = subsample_pixels(args.frame_dir, frames, frames_array, mask)
        pca = PCA(n_components=args.new_size)
        pca = pca.fit(frames_array)

        for i,frame in enumerate(frames):
            frame_array = np.load(args.frame_dir+frame)

            projected_frame = apply_transformation(frame_array,pca)

            save_frame(args.new_frame_dir,projected_frame,i)

    elif(args.mode == 'LDA'):
        frame_subsample_rate = 1 #grab every 'nth' frame
        pixel_prob = .05 #each pixel has a 'k%' chance of being sampled

        labels_shape = np.load(args.label_dir+labels[0]).shape
        frame_shape = np.load(args.frame_dir+frames[0]).shape

        num_frames = len(frames)//frame_subsample_rate
        height = num_frames * frame_shape[0] * frame_shape[1]
        width = frame_shape[2]


        mask = create_rand_mask(height, pixel_prob)

        #Creating array to fit to- of subsampled frames from videos
        frames_array = np.zeros((np.sum(mask),width))
        frames_array = subsample_pixels(args.frame_dir, frames, frames_array, mask)

        labels_array = np.zeros((len(frames_array)))
        labels_array = subsample_labels(args.label_dir, labels, labels_array, mask)

        num_classes = len(set(labels_array))
        lda = LinearDiscriminantAnalysis(n_components=num_classes-1)
        lda = lda.fit(frames_array,labels_array)

        for i,frame in enumerate(frames):
            frame_array = np.load(args.frame_dir+frame)

            projected_frame = apply_transformation(frame_array,lda)

            save_frame(args.new_frame_dir,projected_frame,i)

    # all other dim reduction techniques
    else:
        for frame in frames:
            start = time.time()
            frame_array = np.load(args.frame_dir+frame)


            start = time.time()
            if(args.mode == 'random_projection'):
                projected_frame = gen_random_projection(frame_array, args.new_size)
            if(args.mode == 'gen_labels'):
                projected_frame = make_labels(frame_array)
            if(args.mode == 'SVD'):
                if(i==0):
                    plot_svd(frame_array)



            print(projected_frame.shape)

            save_frame(args.new_frame_dir,projected_frame,i)
            i += 1
