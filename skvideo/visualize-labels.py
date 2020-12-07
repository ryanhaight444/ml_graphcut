"""
Simple tool to show what a label file looked like. Those generated by the main script aren't always great.
"""

import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import skvideo.io
import argparse
from mask import get_mask

# CLI arguments
parser = argparse.ArgumentParser(description='Visualize stuff')
parser.add_argument("labels", help="Path to the labels checkpoint")
parser.add_argument("video", nargs="?", help="Path to the source video for an overlay")
parser.add_argument("-S", "--scale-factor", type=int, default=1,
                    help="If the video is bigger than the labels, this is by how much")
parser.add_argument("--alpha", "-a", type=float, default=0.5,
                    help="Transparency of the labels over the first frame (default mode only)")
parser.add_argument("--cmap", default="jet",
                    help="The color map to use for the embeddings (either mode)")
parser.add_argument("--use-mask", action="store_true")
parser.add_argument("--save-image", action="store_true",
                    help="Save the generated plot as a PNG image")

def show_lables(labels, cmap="gray", save_image=False):
    # Hard-coded dimensions for now. Just load the video in other cases
    labels_grid = np.reshape(labels, (180, 320))
    plt.imshow(labels_grid, cmap=cmap)
    fig = plt.gcf()
    plt.show()
    if save_image:
        fig.savefig(get_label_name(), bbox_inches="tight", pad_inches=0)


def show_overlay(labels_grid, frame, cmap="gray", save_image=False):
    plt.imshow(frame, cmap="gray")
    plt.imshow(labels_grid, cmap=cmap, alpha=args.alpha)
    fig = plt.gcf()
    plt.show()
    if save_image:
        fig.savefig(get_label_name(True), bbox_inches="tight")


def scale_labels(labels, M, N, scale_factor=2):
    orig_shape = (int(M/scale_factor), int(N/scale_factor))
    label_grid = np.reshape(labels, orig_shape)
    upscaled_labels = np.kron(label_grid, np.ones((scale_factor, scale_factor))).astype("int")
    return upscaled_labels


def get_label_name(overlay=False):
    # Remove directory and file extension
    lbl_fn = args.labels[args.labels.rfind("/")+1:-4]
    masked = "-masked" if args.use_mask else ""
    return "labels/{}{}{}.png".format("overlay_" if overlay else "", lbl_fn, masked)


def show_with_mask(labels_grid, mask):
    # Mask out labels and setup new color map
    masked_labels = np.ma.array(labels_grid, mask=np.logical_not(mask))
    cmap = matplotlib.cm.hot
    cmap.set_bad("white", 1.)

    # Setup and show the figure
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(masked_labels, cmap=cmap)
    fig = plt.gcf()
    plt.show()
    if args.save_image:
        fig.savefig(get_label_name(False), bbox_inches="tight")

if __name__ == "__main__":
    args = parser.parse_args()
    cmap = args.cmap
    labels = np.load(args.labels)
    scale = args.scale_factor
    mask = None

    if args.video:
        # Load video and reshape labels
        frame = skvideo.io.vread(args.video, num_frames=1)[0]
        M, N, _ = frame.shape

        # Upscale the labels if needed
        if scale > 1:
            upscaled = scale_labels(labels, M, N, scale)
            labels_grid = upscaled
        else:
            labels_grid = np.reshape(labels, (M, N))

        # Display the image, with or without masking out the background
        if args.use_mask:
            mask = get_mask(skvideo.io.vread(args.video))
            show_with_mask(labels_grid, mask)
        else:
            show_overlay(labels_grid, frame, args.cmap, save_image=args.save_image)
    else:
        show_lables(labels, cmap=cmap, save_image=args.save_image)
