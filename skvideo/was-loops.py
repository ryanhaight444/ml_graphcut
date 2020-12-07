from statsmodels import robust
from collections import namedtuple
import time as sys_time
import datetime
import numpy as np
import skimage.io
import skvideo.io
import skvideo.utils
import gco
from math import gcd
from tqdm import tqdm
import argparse
import helpers
import embeddings_helper
import lookup_table
from mask import get_mask

# CONSTANTS
LAMBDA_T = 400
LAMBDA_S = 100
LAMBDA_STATIC = 100
BETA_DEFAULT = 10

# TYPE DEFINITIONS
TPair = namedtuple("TPair", "start,period")
Coord = namedtuple("Coord", "i,j")

# CLI arguments
parser = argparse.ArgumentParser(description='Graph cuts and such for loops')
parser.add_argument("-e", "--embeddings",
                    help="Name of the embeddings (e.g. '128_projected')")
parser.add_argument('--small-size', action='store_true',
                    help='Only run computation on a small part of the video')
parser.add_argument('--small-time', action='store_true',
                    help='Only run computation on a few frames of the video')
parser.add_argument('-o', '--img-file',
                    help="Where to store the output image")
parser.add_argument("-i", "--input-file",
                    help="Which video file to process")
parser.add_argument("--double-size-input-video",
                    help="The path to the same input video with w'=2*w, h'=2*h")
parser.add_argument("-B", "--beta", type=int, default=BETA_DEFAULT,
                    help="Adjust the multiplier for the smoothness cost")
parser.add_argument("--theta", type=float, default=0.5,
                    help="The relative weight of the embeddings over RGB values. Range [0, 1]")
parser.add_argument("-P", "--periods", type=int, nargs="+", default=-1,
                    help="Manually pick the periods to try")
parser.add_argument("--no-static", action="store_true",
                    help="If included, don't include static pixels in Phase II")
parser.add_argument("-t", "--temp", "--tmp", action="store_true",
                    help="Store files in root dir (temp for testing)")
parser.add_argument("--tag",
                    help="Include this string in output file names")
parser.add_argument("--no-cache", action="store_true",
                    help="Do not read or write cache/checkpoint files")
args = parser.parse_args()

# RUNTIME VARIABLES
beta = args.beta
theta = args.theta
if theta < 0 or theta > 1: theta = 0.5
filename = args.input_file
USE_EMBEDDINGS = True if args.embeddings else False
NO_STATIC = args.no_static


def lcm(a, b):
    return a*b // gcd(a, b)


def cCoord(i, j):
    return Coord(i=i, j=j)


def cTPair(s, p):
    return TPair(start=s, period=p)


def is_static(lbl):
    return lbl.period == 1


def same_period(tp_x, tp_z):
    return tp_x.period == tp_z.period


def V_color(loc, t):
    return vid_color[t % T, loc.i, loc.j]


def distance(x, xframe, z, zframe):
    a = V_color(x, xframe)
    b = V_color(z, zframe)
    rgb = np.linalg.norm(b - a)

    if USE_EMBEDDINGS and theta > 0:
        emb = embeddings_helper.distance(x, xframe, z, zframe)
        return theta * emb + (1-theta) * rgb
    else:
        return rgb


def unary_cost(loc, lbl):
    s = lbl.start
    p = lbl.period

    if not is_loopable[loc.i, loc.j]:
        return 10

    if is_static(lbl):
        return 50000 # TEMP
        # return 0
    d_rgb = np.linalg.norm(V_color(loc, s) - V_color(loc, s + p)) ** 2 + \
            np.linalg.norm(V_color(loc, s-1) - V_color(loc, s + p - 1)) **2
    discon = d_rgb
    if USE_EMBEDDINGS:
        d_emb = embeddings_helper.distance(loc, s, loc, s+p) ** 2 + \
                embeddings_helper.distance(loc, s-1, loc, s + p - 1) ** 2
        discon = (theta * d_emb) + ( (1-theta) * d_rgb)
    succ_var = np.empty(T-1, dtype='int')
    for t in range(T-1):
        # rgb = np.linalg.norm(V_color(loc, t) - V_color(loc, t-1))
        # emb = embeddings_helper.distance(loc, t, loc, t-1)
        succ_var[t] = distance(loc, t, loc, t-1)
    factor = 1 / (1 + LAMBDA_T * robust.mad(succ_var))
    return discon * factor


def gen_pairwise_cost_fn(current_period):
    _period = current_period
    def pairwise_cost(node_x, node_z, start_x, start_z):
        """
        Calculate the pairwise cost/smoothness/spacial consistency term for a given period
        :param node_x:
        :param node_z:
        :param start_x:
        :param start_z:
        :return:
        """
        # return 1 # TEMP
        tp_x = cTPair(start_x, _period)
        tp_z = cTPair(start_z, _period)
        x = id_to_loc(node_x)
        z = id_to_loc(node_z)

        psi_val = psi(x, z, tp_x, tp_z)
        gam_val = gamma(x, z)
        ret = psi_val * gam_val * beta
        return ret
    return pairwise_cost


def gen_phi(lbl):
    s = lbl.start
    p = lbl.period

    def f(t):
        return s + ((t - s) % p)
    return f


# TODO: we could speed this up, I'm sure. Don't need to calculate all 'T' values each time
def gamma(x, z):
    changes = np.empty(T)
    for t in range(T):
        changes[t] = np.linalg.norm(V_color(x, t) - V_color(z, t))
    mad = robust.mad(changes)
    return 1 / (1 + (LAMBDA_S * mad))


def naive_psi(x, z, tp_x, tp_z):
    _start = sys_time.perf_counter()
    sum = 0
    phi_x = gen_phi(tp_x)
    phi_z = gen_phi(tp_z)
    lcm_T = lcm(tp_x.period, tp_z.period)
    for t in range(lcm_T):
        sum += distance(x, phi_x(t), x, phi_z(t)) + \
               distance(z, phi_x(t), z, phi_z(t))
    ret = (1/lcm_T) * sum
    print("naive: %f\tTook: %.6f seconds" % (ret, sys_time.perf_counter() - _start))
    return ret

temp_last_psi = "psi didn't run yet"
case_count = [0, 0, 0, 0]
def psi(x, z, tp_x, tp_z):
    global temp_last_psi
    if is_static(tp_x) and is_static(tp_z):
        case_count[0] += 1
        # Both pixels are static
        temp_last_psi = "both are static"
        return distance(x, tp_x.start, x, tp_z.start) + \
               distance(z, tp_x.start, z, tp_x.start)
    elif is_static(tp_x):
        case_count[1] += 1
        temp_last_psi = "x is static"
        sum = 0
        phi_z = gen_phi(tp_z)
        for t in range(T - 1):
            sum += distance(x, tp_x.start, x, phi_z(t)) + \
                   distance(z, tp_x.start, z, phi_z(t))
        # sum2 = lookup_table.calc_energy_case2(x, z, tp_x, tp_z, vid_color)
        # print("x static; sum: %f\t(1/T)sum: %f\tsum2: %f\t(1/T)sum2: %f" % (sum, (1/T) * sum, sum2, (1/T) * sum2))
        #print("\t", x, z, tp_x, tp_z)
        #assert (1/T) * sum == sum2 or sum == sum2
        return (1 / T) * sum
        # return (1 / T) * sum2
    elif is_static(tp_z):
        case_count[1] += 1
        temp_last_psi = "z is static"
        # I'm just adding this because I think it might make sense to have,
        # but it's not in the paper. The way the paper is written, it's
        # possible that the "if only x is static" case should apply irrespective
        # of the actual order of x and z. That is, if ONE of them is static,
        # these computations should take place. Additionally, if only z is
        # static, then (I think) the final case will compute the LCM of the two
        # periods and determine that the LCM is 0 (since the period of z is 0).
        # This will lead to a division by zero error.

        # TODO: this will need to be done individually for all three color
        #       channels if we move away from b/w (black-and-white)
        sum = 0
        phi_x = gen_phi(tp_x)
        for t in range(T-1):
            sum += distance(x, tp_z.start, x, phi_x(t)) + \
                   distance(z, tp_z.start, z, phi_x(t))
        # sum2 = lookup_table.calc_energy_case2(x, z, tp_x, tp_z, vid_color)
        ret = (1 / T) * sum
        return ret
    elif same_period(tp_x, tp_z):
        case_count[2] += 1
        temp_last_psi = "both have same period"
        sum = 0
        phi_x = gen_phi(tp_x)
        phi_z = gen_phi(tp_z)
        for t in range(0, tp_x.period - 1):
            sum += distance(x, phi_x(t), x, phi_z(t)) + \
                   distance(z, phi_x(t), z, phi_z(t))
        ret = sum / tp_x.period
        return ret
    else:
        case_count[3] += 1
        temp_last_psi = "x, z have diff periods"
        # TODO - does this need to be rewritten? And does it include the error in the paper? (See Errata on Hoppe's page)
        sum = 0
        lcm_T = lcm(tp_x.period, tp_z.period)
        phi_x = gen_phi(tp_x)
        phi_z = gen_phi(tp_z)
        # I think that this is right, but the paper is not explicitly clear
        _start = sys_time.perf_counter()
        for t in range(0, lcm_T - 1):
            sum += distance(x, phi_x(t), x, phi_z(t)) + \
                   distance(z, phi_x(t), z, phi_z(t))
        if lcm_T is 0:
            print("WARNING! LCM was 0!!")
            print(x, z, tp_x, tp_z)
        ret = sum / lcm_T
        # print("case4:", ret)
        #assert np.isclose(ret, fancy)
        # return fancy
        return ret


def get_unary_cost_map(nnodes, period):
    unary = np.empty((nnodes, period))
    for node in tqdm(range(nnodes)):
        loc = id_to_loc(node)
        for s in range(period):
            lbl = cTPair(s, period)
            unary[node, s] = unary_cost(loc, lbl)
    return unary


#
#
# Done with functions
#
#

print("Loading video %s..." % filename)
num_frames = 0
vid = skvideo.io.vread(filename, num_frames=num_frames,
                       outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
vid_color = skvideo.io.vread(filename, num_frames=num_frames)
frame_rate = skvideo.io.ffprobe(filename)['video']['@avg_frame_rate']
T, M, N = vid.shape
if args.small_size:
    M = min(M, 150)
    N = min(N, 150)
if args.small_time:
    T = min(T, 149)
width = N
is_loopable = get_mask(vid_color)

np.set_printoptions(precision=2)

if USE_EMBEDDINGS:
    embeddings_helper.setup(args, T) # TODO: something's off - shouldn't need all the frames
    frames = embeddings_helper.export_frames()

lookup_table.compute_lookup_table(vid_color)

def id_to_loc(id):
    if id < 0: raise Exception("ID must be >= 0")
    i = id // width
    j = id % width
    if i >= M or j >= N:
        raise Exception("Node ID {} too big coord would be ({}, {})".format(id, i, j))
    return cCoord(i, j)

def loc_to_id(loc):
    i, j = loc
    return i*width + j

def save_label_images(labels, period):
    # Store the labels as an image
    label_grid = np.reshape(labels, (M, N))
    imfn_labels_bw = helpers.get_img_fn(args, period, color=False)
    imfn_labels_c = helpers.get_img_fn(args, period, color=True)
    print("Saving labels as images:", imfn_labels_bw, imfn_labels_c)
    img_bw = (label_grid.astype(float) / np.max(labels)) * 255
    img_bw = img_bw.astype("uint16")
    skimage.io.imsave(imfn_labels_bw, img_bw)
    skimage.io.imsave(imfn_labels_c, helpers.gen_label_overlay(vid_color[0], label_grid))


def save_label_video(labels, period):
    out_video_fn = helpers.get_loop_fn(args, period)
    # out_video_lbl_fn = helpers.get_loop_fn(args, period, notes="with-labels")
    print("Writing video in color", out_video_fn)
    outVid = np.empty([period * 2, M, N, 3])
    # outVidLabels = np.empty((period * 2, M, N, 3))
    label_grid = np.reshape(labels, (M, N))
    for t in tqdm(range(period)):
        nodeid = 0
        for i in range(M):
            for j in range(N):
                s = labels[nodeid]
                p = period
                time = s + ((t - s) % p)

                val = V_color(cCoord(i, j), time)
                # Do twice the loop so that output video loops nicely even w/ bad viewer
                outVid[t, i, j] = val
                outVid[t + period, i, j] = val
                # outVidLabels[t, i, j] = val
                # outVidLabels[t + period, i, j] = val

                nodeid += 1

    # Add a red overlay based on the label
    # outVidLabels[..., 0] += label_grid * (255 / np.max(labels))
    # outVidLabels = outVidLabels.astype("uint8")
    skvideo.io.vwrite(out_video_fn, outVid)
    # skvideo.io.vwrite(out_video_lbl_fn, outVidLabels)


def write_video(final_vid_len):
    # Save as a video (or a few)
    out_video_fn = helpers.get_loop_fn(args, period="all")
    print("Writing final loop video to:", out_video_fn)
    outVid = np.empty([final_vid_len, M, N, 3])
    for t in tqdm(range(final_vid_len)):
        nodeid = 0
        for i in range(M):
            for j in range(N):
                tp = gc_lbl_to_tp(nodeid, labels_ph2[nodeid])
                s, p = tp
                # s = labels[nodeid]
                # p = period
                time = s
                # TODO: this was `p > 0`, but I think `p` should never be 0. Double check this
                if p > 1: time += (t-s) % p
                # time = s + ((t - s) % p)

                val = V_color(cCoord(i, j), time)
                outVid[t, i, j] = val

                nodeid += 1

    # Add a red overlay based on the label
    skvideo.io.vwrite(out_video_fn, outVid)
    print("Wrote video", out_video_fn)


def write_2x_video(final_vid_len):
    vid2x = skvideo.io.vread(args.double_size_input_video)
    T2, M2, N2, ch2 = vid2x.shape
    outVid2x = np.empty([final_vid_len, M2, N2, 3])
    # labels_2x = labels_ph2.repeat(2, axis=0).repeat(2, axis=1)
    out_video_fn_2x = helpers.get_loop_fn(args, period="all", notes="2x")
    print("Writing double-size file to:", out_video_fn_2x)
    for t in tqdm(range(final_vid_len)):
        nodeid = 0
        for i in range(M):
            for j in range(N):
                tp = gc_lbl_to_tp(nodeid, labels_ph2[nodeid])
                s, p = tp
                time = s
                # TODO: this was `p > 0`, but I think `p` should never be 0. Double check this
                if p > 1: time += (t-s) % p
                # time = s + ((t - s) % p)

                outVid2x[t, i*2, j*2] = vid2x[time, i*2, j*2]
                outVid2x[t, i*2+1, j*2] = vid2x[time, i*2+1, j*2]
                outVid2x[t, i*2, j*2+1] = vid2x[time, i*2, j*2+1]
                outVid2x[t, i*2+1, j*2+1] = vid2x[time, i*2+1, j*2+1]

                nodeid += 1
    skvideo.io.vwrite(out_video_fn_2x, outVid2x)


def try_get_checkpoints(ckpt_name):
    labels = None
    if args.no_cache:
        return labels
    try:
        labels = np.load(ckpt_name)
        print("Loaded labels from cached file (%s)" % ckpt_name)
    except (IOError, FileNotFoundError) as e:
        print("No cached labels with the given parameters")
    return labels


def PHASE_I(period):
    gc = gco.GCO()
    # M*N nodes in graph, and start frame from 0..period
    # TODO: the number of possible start frames shouldn't depend on period - it should always be T, right?
    gc.create_general_graph(M*N, period)
    print("Setting up neighborhoods")
    for i in range(M):
        for j in range(N):
            here = cCoord(i, j)
            down = cCoord(i+1, j)
            rite = cCoord(i, j+1)
            idh = loc_to_id(here)
            idd = loc_to_id(down)
            idr = loc_to_id(rite)
            if i < M-1 and is_loopable[down.i, down.j]:
                # Add neighbor below us
                gc.set_neighbor_pair(idh, idd, 1)
            if j < N-1 and is_loopable[rite.i, rite.j]:
                # Add neighbor to the right of us
                gc.set_neighbor_pair(idh, idr, 1)

    print("Generating and setting unary costs")
    unary_costs = get_unary_cost_map(M*N, period)
    gc.set_data_cost(unary_costs)

    gc.set_smooth_cost_function(gen_pairwise_cost_fn(period))

    # Run a-b expansion
    print("Running alpha-beta expansion. Started at", datetime.datetime.now())
    gc.expansion(niters=1)

    print("Finished at %s. ENERGY: %d (data: %d; smooth: %d)." %
          (datetime.datetime.now(), gc.compute_energy(), gc.compute_data_energy(), gc.compute_smooth_energy()))

    return gc.get_labels()


def calculate_or_load_phase_I(period):
    global case_count
    ckpt_name = helpers.get_checkpoint_name("labels", args, period)

    labels = try_get_checkpoints(ckpt_name)
    checkpoint_exists = labels is not None
    if not checkpoint_exists:
        start = sys_time.time()
        labels = PHASE_I(period)
        print("phase 1 for this period took %.2f seconds" % (sys_time.time() - start))

        # Store it as a numpy object for next time
        if not args.no_cache:
            np.save(ckpt_name, labels)

        print("()" * 30)
        print("case count:", case_count)
        case_count = [0, 0, 0, 0]

        save_label_images(labels, period)
        save_label_video(labels, period)
    return labels


def all_phase_I(periods):
    for index, period in enumerate(periods):
        print("-" * 10, "Period:", period, "-" * 10)

        # Store the labels for this period to be used later
        labels_ph1[index] = calculate_or_load_phase_I(period)


# Paper says: min of 32 frames and go by intervals of 4 frames
periods = args.periods if args.periods is not -1 else range(32, T, 4)
labels_ph1 = [np.zeros((M*N), dtype="int32")] * len(periods)
all_phase_I(periods)


# TODO: implement optimization for pixel diffing that we talked about recently

def gc_lbl_to_tp(nodeid, label):
    if args.no_static:
        _s = labels_ph1[label][nodeid]
        _p = periods[label]
    elif label < T:
        _s = label
        _p = 1
    elif label < T + len(periods):
        _s = labels_ph1[label - T][nodeid]
        _p = periods[label - T]
    else:
        raise Exception("Label {} is too big!".format(label))
    return cTPair(_s, _p)


tmp_case_ct = 0
def pairwise_cost_p2(node_x, node_z, gc_lbl_x, gc_lbl_z):
    global tmp_case_ct
    tp_x = gc_lbl_to_tp(node_x, gc_lbl_x)
    tp_z = gc_lbl_to_tp(node_z, gc_lbl_z)
    x = id_to_loc(node_x)
    z = id_to_loc(node_z)

    psi_val = psi(x, z, tp_x, tp_z)
    gam_val = gamma(x, z)
    ret = psi_val * gam_val * beta
    # if ret < 1:
    #     print("pairwise_cost_p2: rounding up to 1 from", ret)
    #     ret = max(ret, 1)
    tmp_case_ct += 1
    # if (tmp_case_ct % 10000 == 0) and np.sum(case_count[1:]):
    #     print(case_count)
    return ret


def get_unary_cost_map_ph2(nnodes, num_labels):
    unary = np.empty((nnodes, num_labels))
    for node in tqdm(range(nnodes)):
        loc = id_to_loc(node)
        for t in range(num_labels):
            tp = gc_lbl_to_tp(node, t)
            unary[node, t] = unary_cost(loc, tp)
    return unary


# Create the final graph to optimize
def calc_phase_ii(num_labels):
    global temp_last_psi
    gc = gco.GCO()
    gc.create_general_graph(M*N, num_labels)
    print("Setting up neighborhoods")
    for i in range(M):
        for j in range(N):
            here = cCoord(i, j)
            down = cCoord(i+1, j)
            rite = cCoord(i, j+1)
            idh = loc_to_id(here)
            idd = loc_to_id(down)
            idr = loc_to_id(rite)
            if i < M-1 and is_loopable[down.i, down.j]:
                # Add neighbor below us
                gc.set_neighbor_pair(idh, idd, 1)
            if j < N-1 and is_loopable[rite.i, rite.j]:
                # Add neighbor to the right of us
                gc.set_neighbor_pair(idh, idr, 1)

    print("Generating and setting unary costs")
    unary_costs = get_unary_cost_map_ph2(M*N, num_labels)
    gc.set_data_cost(unary_costs)

    gc.set_smooth_cost_function(pairwise_cost_p2)

    # Run a-b expansion
    print("Running alpha-beta expansion. Started at", datetime.datetime.now())
    try:
        gc.expansion(niters=1)
    except Exception as e:
        print("-" * 80)
        print("EXCEPTION in gc.expansion!")
        print("psi last value:", temp_last_psi)
        print(e)
        print("Ignoring and getting labels and moving on and such")

    print("Finished at %s. ENERGY: %d (data: %d; smooth: %d)." %
          (datetime.datetime.now(), gc.compute_energy(), gc.compute_data_energy(), gc.compute_smooth_energy()))
    print(case_count)

    return gc.get_labels()


def phase_ii():
    print("")
    print("-" * 10, "Starting final step (combining graphs)", "-" * 10)
    print("")

    # Make sure we consider static pixels in this part ==> T + len(periods)
    num_labels = len(periods) if args.no_static else T + len(periods)

    ckpt_name = helpers.get_checkpoint_name("labels", args, period="all")

    labels = try_get_checkpoints(ckpt_name)
    if labels is None:
        start = sys_time.time()
        labels = calc_phase_ii(num_labels)
        print("phase 2 took %.2f seconds" % (sys_time.time() - start))
        print("()" * 30)
        print("case count:", case_count)

        # Store it as a numpy object for next time
        if not args.no_cache:
            np.save(ckpt_name, labels)

    save_label_images(labels, "all")
    return labels

# now, we do phase 2
# Want to find the best period from all possible periods (`periods` array)
# The cost of selecting a given period is the cost of using that period and
#   the associated start time with that period at that pixel
labels_ph2 = phase_ii()


lcm = np.lcm.reduce(periods)
write_video(lcm)

if args.double_size_input_video:
    write_2x_video(args.double_size_input_video)
