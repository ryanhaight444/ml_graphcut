import os
import datetime
import numpy as np
import skimage.color

RESULTS_BASE_DIR = "/projects/wehrresearch/ml_graphcut/results/"
# When we make breaking changes, we can't load the old checkpoints. Change this to invalidate old checkpoints.
VERSION = "04"

def results_dir(cli_args):
    TEMP_DIR = "temp"
    if cli_args.temp:
        if not os.path.isdir(TEMP_DIR):
            os.mkdir(TEMP_DIR)
        return TEMP_DIR

    now = datetime.datetime.now()
    today = "{}-{:02}-{:02}".format(now.year, now.month, now.day)
    p = RESULTS_BASE_DIR + today
    if not os.path.isdir(RESULTS_BASE_DIR):
        os.mkdir(RESULTS_BASE_DIR)
    if not os.path.isdir(p):
        os.mkdir(p)
    return p

def experiment_details_string(cli_args, period, theta, notes=''):
    beta = cli_args.beta
    tag = cli_args.tag or ""
    if tag: tag = "_" + tag
    embed = ""
    if cli_args.embeddings:
        embed = "_" + cli_args.embeddings + "-T{:.2f}".format(theta)
    if notes: notes = "_" + notes.replace(" ", "-")
    if cli_args.no_temporal_embeddings:
        notes += "_NTE"
    # if cli_args.restrict_embeddings_to is not None:
    #     notes = notes + "_emb-" + cli_args.restrict_embeddings_to + "-only"
    return "v{}_B{}_P{}{}{}{}".format(VERSION, beta, period, embed, notes, tag)



def get_fn(name, cli_args, period, theta):
    p = results_dir(cli_args)
    rest = experiment_details_string(cli_args, period, theta)
    return "{}/{}_{}".format(p, name, rest)

def get_loop_fn(cli_args, period, theta, notes=''):
    base = experiment_details_string(cli_args, period, theta, notes)
    filename = cli_args.input_file
    video_name = filename[filename.rfind("/") + 1: filename.rfind(".")]
    p = results_dir(cli_args)
    name = "{}_{}.mp4".format(video_name, base)
    return p + "/" + name

def get_img_fn(cli_args, period, theta, color=False, notes=''):
    base = experiment_details_string(cli_args, period, theta, notes)
    p = results_dir(cli_args)
    c = "color" if color else "bw"
    vid_name = get_video_name(cli_args)
    if cli_args.img_file:
        name = cli_args.img_file
        if base.endswith(".png"): name = name[:-4]
    else:
        name = "labels-{}_{}_{}.png".format(c, vid_name, base)
    return p + "/" + name


def get_checkpoint_name(name, cli_args, period, theta):
    DIR = "checkpoints"
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    base = experiment_details_string(cli_args, period, theta)
    video_name = get_video_name(cli_args)
    return "{}/{}_{}_{}.npy".format(DIR, name, video_name, base)


def get_video_name(cli_args):
    video_path = cli_args.input_file
    idx = 0
    try: idx = video_path.rindex("/") + 1
    except Exception: pass
    return video_path[idx:]


def get_log_fn(name, theta, cli_args):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H%M")
    vid_name = get_video_name(cli_args)
    metric = "rgb"
    if cli_args.embeddings:
        metric = "{}x{}".format(theta, cli_args.embeddings)
    return "logs/{}_{}_{}_{}".format(now, vid_name, metric, name)

# CREATING COLORED IMAGES TO VISUALIZE LABELS

def gen_label_overlay(frame, labels, alpha=.2):
    assert(len(frame.shape) is 3)
    # Scale the labels so they range from 0 to 255
    mask = (0-labels) * (255 / np.max(labels))

    # Set up greyscale image
    gray = skimage.color.rgb2gray(frame) * 255
    output = np.empty(frame.shape)
    output[...,0] = gray * alpha + (1 - alpha) * mask
    output[...,1] = gray
    output[...,2] = gray
    output = output.astype("uint8")
    return output

