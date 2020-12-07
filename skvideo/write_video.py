import skvideo.io
from scipy.ndimage import gaussian_filter
import numpy as np
import mask
from tqdm import tqdm
disable_tqdm = False
do_blend_t = False
do_blend_s = False

def write_video(source, time_map, out_fn, loop_len=-1, repeat=1, loopable_grid=None):
    T, M, N, ch = source.shape
    vid_len = loop_len
    if vid_len < 0:
        vid_len = time_map.shape[0]
        print("Defaulting single loop length to", vid_len)
    total_vid_len = vid_len * repeat
    outVid = np.empty((total_vid_len, M, N, ch))
    time_map = time_map.astype("uint8")

    # Build up the first iteration of the video using the time map
    print("Building first pass based on time_map")
    for t in tqdm(range(vid_len), disable=disable_tqdm):
        for i in range(M):
            for j in range(N):
                time = time_map[t, i, j]
                outVid[t, i, j] = source[time, i, j]
    outVid = outVid.astype("uint8")
    print(outVid.shape, outVid.size)
    assert np.max(outVid) > 0

    if do_blend_t:
        # Now we blend the video based on when the video loops
        restarts = get_restarts(time_map)
        print("Temporal blending")
        for t in tqdm(range(vid_len), disable=disable_tqdm):
            for i in range(M):
                for j in range(N):
                    # Temporal blending over loops
                    consider = loopable_grid[i, j] if loopable_grid is not None else False
                    if restarts[t, i, j] and consider:
                        s = t-2
                        e = t+3
                        for ch in range(3):
                            indices = range(s, e)
                            original = source.take(indices, axis=0, mode="wrap")
                            blended = np.matmul(get_filter(), original[:, i, j, ch])
                            # If this crashes with an IndexOutOfBounds error, it's prob b/c loop restart is near the edge
                            # of the video. The fix would be something like:
                            # for i in range(len(indicies)):
                            #     outVid[s+i % vid_len, i, j, ch] = blended[i]
                            # or something like that. Slower than this so I'm leaving it for now
                            outVid[s:e, i, j, ch] = blended

    if do_blend_s:
        print("Spatial blending (if it works?)")
        for t in tqdm(range(vid_len), disable=disable_tqdm):
            mask = np.zeros((M, N))
            for i in range(M-1):
                for j in range(N-1):
                    # Mask[i, j] is true if pixel to bottom or right has different time map
                    mask[i, j] = time_map[t, i, j] != time_map[t, i+1, j] or time_map[t, i, j] != time_map[t, i, j+1]
            if np.max(mask) > 0:
                # There is some blending to do. Expand the size of the mask and normalize it
                mask = gaussian_filter(mask, sigma=10)
                mask /= np.max(mask)

                # Now blur the whole frame and copy the pixels near the seams into the output
                blurred = gaussian_filter(outVid[t], sigma=5)
                outVid[t] = (mask[...,np.newaxis] * blurred) + ((1 - mask[..., np.newaxis]) * outVid[t])
                # outVid[t, mask.astype("bool"), 0] = 255

    # Repeat it as needed (helpful for when video players jump when looping a video)
    print("Copying repeats")
    for rep in range(1, repeat):
        s = vid_len*rep
        e = vid_len*(rep+1)
        outVid[s:e] = outVid[:vid_len]

    print("Writing final video to", out_fn)
    writer = skvideo.io.FFmpegWriter(out_fn)
    for t in tqdm(range(total_vid_len)):
        writer.writeFrame(outVid[t])
    writer.close()
    # skvideo.io.vwrite(out_fn, outVid)

def write_HD_P2(source, small_time_map, kron_shape, out_fn, vid_len):
    T, M, N, ch = source.shape
    outVid = np.empty((vid_len, M, N, ch))
    print("Building first pass based on time_map")
    for t in tqdm(range(vid_len), disable=disable_tqdm):
        time_map_t = np.kron(small_time_map[t], np.ones(kron_shape)).astype("int")
        for i in range(M):
            for j in range(N):
                time = time_map_t[i, j]
                outVid[t, i, j] = source[time, i, j]
    print("Writing final video to", out_fn)
    writer = skvideo.io.FFmpegWriter(out_fn)
    for t in tqdm(range(vid_len)):
        writer.writeFrame(outVid[t])
    writer.close()


def get_restarts(time_map):
    """
    For each pixel of each frame in a video, detemrine where its loop restarts
    :param time_map: A time map, as used above, in the shape (T, M, N)
    :return: A matrix of size (T, M, N) with 0 in most entries, 1 where the loop repeats
    """
    restarts = np.zeros(time_map.shape)
    T = restarts.shape[0]
    for t in range(T-1):
        restarts[t] = time_map[t] + 1 != time_map[t+1]
    return restarts


def get_filter():
    # How do we write this nicer?
    f = np.array([[1, 0, 0, 0, 0],
                  np.array([1, 1, 1, 0, 0]) * 1.0 / 3,
                  np.ones((5,)) * 1.0 / 5,
                  np.array([0, 0, 1, 1, 1]) * 1.0 / 3,
                  [0, 0, 0, 0, 1]])
    return f


def write_p1_video(source, start_frame_labels, period, out_fn, interval=1, repeat=2, loopable_grid=None):
    T, M, N, _ = source.shape
    # Handling for source videos in range [0..1]. May not need, so could replace with an assert later
    if np.max(source) <= 1:
        source *= 255
        source = source.astype("int")
    vid_len = period * repeat
    labels_grid = np.reshape(start_frame_labels, (M, N))
    time_map = np.zeros((vid_len, M, N))
    print("Building p1 time map")
    for t in tqdm(range(period), disable=disable_tqdm):
        for i in range(M):
            for j in range(N):
                s = labels_grid[i, j] * interval
                p = period
                time = s + ((t - s) % p)
                if (loopable_grid is not None) and (not loopable_grid[i, j]):
                    time = 0 # for now
                    # time = s

                time_map[t, i, j] = time

    # for rep in range(1, repeat):
    #     s = period*rep
    #     e = period*(rep+1)
    #     outVid[s:e] = outVid[:period]
    #
    # skvideo.io.vwrite(out_fn, outVid)
    write_video(source, time_map, out_fn, period, repeat, loopable_grid=loopable_grid)


def write_p1_upscaled_video(large_source, labels, period, out_fn, interval=1, repeat=2, loopable_grid=None):
    T, vM, vN, _ = large_source.shape
    lM, lN = 180, 320
    # Make sure our dimensions match up
    assert vM/lM == vN/lN, "aspect ratio of video doesn't match that of the labels"

    label_grid = np.reshape(labels, (lM, lN))
    scale_ratio = int(vM / lM)
    upscaled_labels = np.kron(label_grid, np.ones((scale_ratio, scale_ratio))).astype("int")
    write_p1_video(large_source, np.ravel(upscaled_labels), period, out_fn, interval, repeat, loopable_grid)

def write_upscaled_video_from_time_map(large_source, time_map, out_fn, repeat=2):
    print("writing from a time_map checkpoint")
    vT, vM, vN, _ = large_source.shape
    mT, mM, mN = time_map.shape
    assert vM/mM == vN/mN, "aspect ratios not same between time map and source video"

    scale_ratio = int(vM / mM)
    upscaled_time_map = np.kron(time_map, np.ones((1, scale_ratio, scale_ratio))).astype("int")
    write_video(large_source, upscaled_time_map, out_fn, repeat=repeat)


if __name__ == "__main__":
    import argparse
    import re
    parser = argparse.ArgumentParser(description='Write a phase 1 video given the labels')
    parser.add_argument("source",
                        help="The source video")
    parser.add_argument("labels_or_time_map",
                        help="The labels or time_map (npy checkpoint file)")
    parser.add_argument("-P", "--period", type=int,
                        help="The period associated with these labels")
    parser.add_argument("-R", "--repeat", type=int, default=2,
                        help="Number of times to repeat the loop in the output video")
    parser.add_argument("--upscaled", "-u", action="store_true",
                        help="Use if the 'source' video is larger dimensions than the labels")
    parser.add_argument("-i", "--interval", type=int, default=4,
                        help="The interval between start frames considered in the original graph cut")
    parser.add_argument("--temporal-blending", action="store_true")
    parser.add_argument("--spatial-blending", action="store_true")
    parser.add_argument("-o", "--out", help="The output video name")

    args = parser.parse_args()

    # Misc arg parsing stuff
    # loopable = mask.get_mask(src)
    loopable = None
    disable_tqdm = False
    do_blend_t = args.temporal_blending
    do_blend_s = args.spatial_blending

    # Load video and checkpoint
    src = skvideo.io.vread(args.source)
    cp_name = args.labels_or_time_map
    labels = None
    tm = None
    prefix = ""
    checkpoint = np.load(cp_name)
    if "labels_" in cp_name:
        labels = checkpoint
        prefix = "labels_"
    elif "labels-" in cp_name:
        labels = checkpoint
        prefix = "labels-"
    elif "time_map_" in cp_name:
        tm = checkpoint
        prefix = "time_map_"
    else:
        print("Unknown type of checkpoint. Must include 'labels_' or 'labels-' or 'time_map_' in file name")
        exit(1)

    # Figure out a default name for the output file
    fn = args.out #or "temp-output.mp4"
    if not fn:
        idx_start = cp_name.rfind(prefix)
        idx_ext = cp_name.rfind(".")
        additional = ""
        if do_blend_t: additional += "-temporal"
        if do_blend_s: additional += "-spatial"
        fn = cp_name[idx_start+len(prefix):idx_ext] + additional + ".mp4"
        print("saving to", fn)

    period = args.period
    if labels is not None and not args.period:
        pattern = "_P(\d+)_"
        matches = re.findall(pattern, cp_name)
        if len(matches) > 0:
            period = int(matches[0])
            print("Assuming period of %d" % period)

    if args.upscaled:
        if labels is not None:
            write_p1_upscaled_video(src, labels, period, interval=args.interval, out_fn=fn, repeat=args.repeat, loopable_grid=loopable)
        else:
            write_upscaled_video_from_time_map(src, tm, out_fn=fn, repeat=args.repeat)
    else:
        if labels is None:
            print("Not implemented right now. You must provide an 'upscaled' video")
            exit(1)
        write_p1_video(src, labels, period, interval=args.interval, out_fn=fn, repeat=args.repeat, loopable_grid=loopable)
