import os
import numpy as np
from tqdm import tqdm
import helpers

embeddings_dir = "SETUP_NOT_CALLED_IN_EMBEDDINGS_HELPER"
frames = []

EMBEDDINGS_DIR = "/projects/wehrresearch/ml_graphcut/embeddings"

def setup(cli_args, num_frames, disable_tqdm=False, zero_last_frame=False):
    global embeddings_dir
    global frames
    mult = 1
    if cli_args.fps60: mult = 2
    emb_arg = cli_args.embeddings
    if not emb_arg:
        # We're not using embeddings at all
        return

    video_name = helpers.get_video_name(cli_args)
    embeddings_dir = "{}/{}/{}/".format(EMBEDDINGS_DIR, emb_arg, video_name)

    if not os.path.isdir(embeddings_dir):
        print("")
        print("No such embeddings exist!! (Did you un-tar them? Have they been computed?)")
        print("\tLooked here:", embeddings_dir)
        exit(127)

    print("Loading and normalizing frames from %s" % embeddings_dir)
    shape = (num_frames, ) + load_frame(0).shape
    frames = np.empty(shape)
    for i in tqdm(range(num_frames), disable=disable_tqdm):
        frame = load_frame(i*mult)
        if zero_last_frame:
            frame[..., 0] = np.zeros(frame[..., 0].shape)
        frames[i] = frame
    emb_min = np.min(frames)
    emb_max = np.max(frames)
    frames += 0 - emb_min # Scale up so min is 0
    frames /= (emb_max + ( 0 - emb_min) ) # get into the range 0..1
    frames *= 255 # get into the range 0..255


def load_frame(time):
    name = "frame{:0>6d}.npy".format(time)
    p = "{}{}".format(embeddings_dir, name)
    return np.load(p)


def get_frame(time):
    if time >= len(frames):
        print("requested frame %d but len(frames)=%d" % (time, len(frames)))
    return frames[time]

def export_frames():
    return frames

def distance(x, tx, z, tz):
    frame_X = get_frame(tx)
    frame_Z = get_frame(tz)
    emb_x = frame_X[x.i, x.j]
    emb_z = frame_Z[z.i, z.j]
    ret = np.sum( (emb_z - emb_x) ** 2)
    assert ret >= 0
    return ret
