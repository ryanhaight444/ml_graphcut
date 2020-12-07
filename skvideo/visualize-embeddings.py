import matplotlib.pyplot as plt
import numpy as np
import skvideo.io
import skimage.io
import argparse
from tqdm import tqdm

WEHRRESEARCH="/projects/wehrresearch/ml_graphcut/"
EMB_DIR = WEHRRESEARCH + "embeddings/" # recsnow_01-320x180.mp4/frame000149.npy"
VID_DIR = WEHRRESEARCH + "assets/"

# CLI arguments
parser = argparse.ArgumentParser(description='Visualize stuff')
parser.add_argument("source", help="Path to the source video or image")
parser.add_argument("-e", "--embeddings", help="Name of the embeddings")
parser.add_argument("--distances", "-d", action="store_true",
                    help="Clickable UI that shows distances from a selected pixel")
parser.add_argument("-n", "--num-frames-dist", type=int,
                    help="Show distance to the frame `n` frames away")
parser.add_argument("--step", type=int, default=1,
                    help="How many frames ahead to skip when stepping through time")
parser.add_argument("--max-frames", type=int, default=10,
                    help="The maximum number of embedding frames to load")
parser.add_argument("--alpha", "-a", type=float, default=0.5,
                    help="Transparency of the labels over the first frame (default mode only)")
parser.add_argument("--cmap", default="jet",
                    help="The color map to use for the embeddings (either mode)")
parser.add_argument("--freeze-reference", action="store_true",
                    help="Use the first frame as a reference frame when stepping through time.")
parser.add_argument("--save-images", action="store_true",
                    help="Store an image each time the distance image is clicked")
parser.add_argument("--i-cord", type=int)
parser.add_argument("--j-cord", type=int)

def load_embeddings(video_name, embeddings_name, frame_num=0):
    source = video_name
    slash_idx = source.rfind("/") + 1
    vid_name = source[slash_idx:]
    emb_fn = EMB_DIR + embeddings_name + "/" + vid_name
    emb_frame_name = "frame{:0>6d}.npy".format(frame_num)
    embeddings = np.load(emb_fn + "/" + emb_frame_name)
    # For bad LDA (first frame broken), instead return `embeddings[...,1:]`
    return embeddings


def load_all_embeddings(video_name, embeddings_name):
    print("Loading all embeddings")
    # Figure out how many frames to load
    max_frames = args.max_frames
    step = args.step
    vid_reader = skvideo.io.FFmpegReader(video_name)
    num_frames, M, N, _ = vid_reader.getShape()
    if max_frames is not None: num_frames = max_frames

    # Figure out embedding shapes
    emb_shape = load_embeddings(video_name, embeddings_name).shape
    shape = (num_frames, ) + emb_shape
    embeddings = np.empty( shape )

    # Load 'em
    for i in tqdm(range(num_frames)):
        embeddings[i] = load_embeddings(video_name, embeddings_name, i*step)

    return embeddings


def load(cli_args):
    num_frames = 1
    source = args.source
    embeddings = None
    is_image = source.endswith(".png") or source.endswith(".jpg") or source.endswith(".jpeg")
    if is_image:
        frame = skimage.io.imread(source)[...,0:3]
    else:
        frame = skvideo.io.vread(source, num_frames=num_frames)[0]

    if cli_args.embeddings:
        embeddings = load_embeddings(source, cli_args.embeddings)

    return frame.astype("float") / 255, embeddings


def show_arg_max(first_frame, first_embeddings, cmap):
    am = np.argmax(first_embeddings, axis=2)

    plt.imshow(first_frame, cmap="gray")
    plt.imshow(am, cmap=cmap, alpha=args.alpha)
    plt.show()


def save_image(m, i, j, cmap="gray", n=0):
    additional = ""
    if n:
        additional += "-n{:0>3d}".format(n)
    if args.embeddings:
        additional += "-{}".format(args.embeddings)
    fn = "weights-{}-{}{}.png".format(i, j, additional)
    print("Saving to: ", fn)
    plt.imsave(fn, m, format="png", cmap=cmap)

# FeatVis class originally written by Dr. Scott Wehrwein
# Modifications by Noah Strong, April 2019
mn = 2
nf = 5
class FeatVis(object):

    def __init__(self, image, features, cmap="", save_images=False, comp_feats=None):
        self.img = image
        self.feats = features
        self.cmap = cmap
        self.save_images = save_images
        self.fig, self.axarr = plt.subplots(2)
        self.colorbar = None
        self.frame_num = 0
        self.i = None
        self.j = None
        # To do RGB distances, comment out next two lines and uncomment the third
        self.embeddings = load_all_embeddings(args.source, args.embeddings)
        self.comp_feats = self.embeddings[0]
        # self.comp_feats = image

        step = args.step
        self.step = step
        self.video = skvideo.io.vread(args.source, num_frames=args.max_frames*step or 0)
        if step > 1:
            print(self.video.shape)
            print(self.video[::step].shape)
            self.video = self.video[::step]

        self.axarr[0].imshow(self.img)
        self.fig.canvas.mpl_connect('button_press_event', self)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress)
        plt.show()

    def keypress(self, event):
        skip = 1
        if event.key is "left":
            if self.frame_num - skip < 0:
                # print("Can't go back - at frame", self.frame_num * self.step)
                return
            self.frame_num -= skip
        elif event.key is "right":
            if self.frame_num + skip >= self.embeddings.shape[0]:
                # print("Can't go forwards - at frame", self.frame_num * self.step)
                return
            self.frame_num += skip
        else:
            return

        self.comp_feats = self.embeddings[self.frame_num]
        if not args.freeze_reference:
            self.feats = self.comp_feats
            self.img = self.video[self.frame_num]
        self.draw(event)

    def draw(self, event):
        j, i = self.j, self.i
        print('drawing (%d, %d) at frame %d' % (i, j, self.frame_num*self.step))

        c = self.feats[i,j,:]
        m = np.linalg.norm(self.comp_feats - c, axis=2)

        # Add red highlight where we clicked
        clicked_here = plt.Circle((j, i), 3, color="r")
        self.axarr[0].clear()
        self.axarr[0].imshow(self.img)
        self.axarr[0].add_artist(clicked_here)

        self.axarr[1].clear()
        img = self.axarr[1].imshow(m, cmap=self.cmap)
        clicked_here_2 = plt.Circle((j, i), 2, color="white")
        self.axarr[1].add_artist(clicked_here_2)
        if self.colorbar: self.colorbar.remove()
        self.colorbar = plt.colorbar(img, ax=self.axarr[1])

        # plt.show was causing an infinite recursion error. Per this SO post:
        # https://stackoverflow.com/questions/39296892/how-to-fight-with-maximum-recursion-depth-exceeded-using-matplotlib-pyplot/39299342
        # I changed it to event.canvas.draw(). I guess plt.show() was creating multiple plots and not deleting the old
        # ones? Or something along those lines. Anyhow, it seems happier now.
        event.canvas.draw()

        return m

    def __call__(self, event):
        if event.xdata is None or event.ydata is None: return
        j, i = int(round(event.xdata)), int(round(event.ydata))
        if args.i_cord and args.j_cord:
            j, i = args.j_cord, args.i_cord
        self.i = i
        self.j = j

        m = self.draw(event)

        if self.save_images:
            save_image(m, i, j, cmap=self.cmap, n=args.num_frames_dist)


if __name__ == "__main__":
    args = parser.parse_args()
    cmap = args.cmap

    first_frame, first_embeddings = load(args)
    distances = first_embeddings if args.embeddings else first_frame

    if args.distances:
        FeatVis(first_frame, distances, cmap, args.save_images)
    else:
        show_arg_max(first_frame, distances, cmap)
