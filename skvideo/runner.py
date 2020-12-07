from statsmodels import robust
from collections import namedtuple
import traceback
import time as sys_time
import datetime
import numpy as np
import skimage.io
import skvideo.io
import skvideo.utils
import gco
from math import gcd
from tqdm import tqdm
import helpers
import embeddings_helper
from mask import get_mask
from write_video import write_p1_video, write_video, write_HD_P2
import lookup_table

# CONSTANTS
LAMBDA_T = 400
LAMBDA_S = 100
# LAMBDA_STATIC = 100
BETA_DEFAULT = 10
P1_FRAME_INTERVAL = 4 # Skip this many frames when choosing start frames
P2_FRAME_INTERVAL = 8
DISTANCE_SCALE = 1
LONGEST_P2_VID = 30 * 60 # 30 frames * 60 seconds = 1 minute

# TYPE DEFINITIONS
TPair = namedtuple("TPair", "start,period")
Coord = namedtuple("Coord", "i,j")

def int_lcm(a, b):
    return a*b // gcd(a, b)


def cCoord(i, j):
    return Coord(i=i, j=j)


def cTPair(s, p):
    return TPair(start=s, period=p)


def is_static(lbl):
    return lbl.period == 1


def same_period(tp_x, tp_z):
    return tp_x.period == tp_z.period


def gen_phi(lbl):
    s = lbl.start
    p = lbl.period

    def f(t):
        return s + ((t - s) % p)
    return f


def try_get_checkpoints(checkpoint_name, no_cache=False):
    labels = None
    if no_cache:
        return labels
    try:
        labels = np.load(checkpoint_name)
        print("Loaded labels from cached file (%s)" % checkpoint_name)
    except (IOError, FileNotFoundError) as e:
        pass
    return labels


def p2_period_shortname(periods):
    # l = len(periods)
    # low = periods[0]
    # hi = periods[-1]
    # return "p2-" + periods.join("-")
    return "-".join(str(p) for p in periods)
    # return "n{}-{}-{}".format(l, low, hi)


class Runner:
    def __init__(self, video, periods, args, is_mpi=False, custom_print_fn=None, theta=None):
        # Save video details
        T, M, N, _ = video.shape
        self.video = video
        self.args = args
        self.T = T
        self.M = M
        self.N = N
        self.width = N

        # Save command-line args
        self.beta = args.beta
        # _theta = args.theta
        _theta = theta
        if _theta < 0 or _theta > 1: _theta = 0.5
        self.theta = _theta
        self.static_cost = args.static_cost
        self.USE_EMBEDDINGS = True if args.embeddings else False
        self.NO_STATIC = args.no_static
        self.do_log = args.do_log

        # Setup loopable mask
        self.is_loopable = get_mask(video)

        # LOOKUP TABLE
        lookup_table.compute_lookup_table(video)

        # Other setup
        self.disable_tqdm = is_mpi
        self.periods = periods
        self.is_mpi = is_mpi
        if not is_mpi:
            self.labels_ph1 = [np.zeros((M * N), dtype="int32")] * len(self.periods)
        if custom_print_fn:
            self.print = custom_print_fn
        else:
            self.print = print
        self.logFile = None
        self.tmp_dist_ct = 0

        # Set up embeddings
        if self.USE_EMBEDDINGS:
            embeddings_helper.setup(args, T, disable_tqdm=self.disable_tqdm, zero_last_frame=args.lda_frame_0_broken)
            self.frames = embeddings_helper.export_frames()

    # General Utility Functions

    def initialize_graph(self, M, N, num_labels):
        is_loopable = self.is_loopable
        gc = gco.GCO()
        gc.create_general_graph(M * N, num_labels)
        # self.print("Setting up neighborhoods")
        for i in range(M):
            for j in range(N):
                here = cCoord(i, j)
                down = cCoord(i + 1, j)
                rite = cCoord(i, j + 1)
                idh = self.loc_to_id(here)
                idd = self.loc_to_id(down)
                idr = self.loc_to_id(rite)
                if i < M - 1 and is_loopable[down.i, down.j]:
                    # Add neighbor below us
                    gc.set_neighbor_pair(idh, idd, 1)
                if j < N - 1 and is_loopable[rite.i, rite.j]:
                    # Add neighbor to the right of us
                    gc.set_neighbor_pair(idh, idr, 1)
        return gc

    def V_color(self, loc, t):
        T = self.T
        if t >= T:
            self.print()
            self.print("Tried to get invalid frame! Beyond end of video. t=%d. location:" % t, loc)
            raise Exception("Tried to get invalid frame of video (frame %d of %d)" % (t, T))
        if t == -1:
            t = T-1
            # print("Wraping t=-1 to t=%d", t)
        return self.video[t, loc.i, loc.j]

    def id_to_loc(self, nodeid):
        width = self.width
        M, N = self.M, self.N
        if nodeid < 0: raise Exception("Node ID must be >= 0")
        i = nodeid // width
        j = nodeid % width
        if i >= M or j >= N:
            raise Exception("Node ID {} too big coord would be ({}, {})".format(nodeid, i, j))
        return cCoord(i, j)

    def loc_to_id(self, loc):
        width = self.width
        i, j = loc
        return i * width + j

    def p1_label_to_tp(self, label_id, period):
        start = label_id * P1_FRAME_INTERVAL
        assert start <= self.T - period
        return cTPair(start, period)

    def p1_tp_to_label(self, tp):
        start, period = tp
        assert start % P1_FRAME_INTERVAL == 0
        return start // P1_FRAME_INTERVAL

    def provide_p1_labels_for_period(self, index, labels):
        """
        When running with MPI, we don't need each instance of the runner to save its own copy of the phase_1 labels,
        and of course the root process won't have the labels from the other processes. Thus before calling phase_2(),
        we must 'provide' all the earlier calculations to the root process with this function.
        :param index: idx into the 'periods' array
        :param labels: the labels for the period periods[index]
        :return:
        """
        self.labels_ph1[index] = labels

    def provide_all_p1_labels(self, labels):
        """
        Maybe it's easier to plug in all the labels at once. Here's how we'd do that.
        :param labels:
        :return:
        """
        self.labels_ph1 = labels

    def save_label_images(self, labels, period):
        M, N = self.M, self.N
        # Store the labels as an image
        label_grid = np.reshape(labels, (M, N))
        imfn_labels_bw = helpers.get_img_fn(self.args, period, color=False)
        imfn_labels_c = helpers.get_img_fn(self.args, period, color=True)
        self.print("Saving labels as images:", imfn_labels_bw, imfn_labels_c)
        img_bw = (label_grid.astype(float) / np.max(labels)) * 255
        img_bw = img_bw.astype("uint16")
        skimage.io.imsave(imfn_labels_bw, img_bw)
        skimage.io.imsave(imfn_labels_c, helpers.gen_label_overlay(self.video[0], label_grid))

    def save_p1_video(self, labels, period):
        out_video_fn = helpers.get_loop_fn(self.args, period, self.theta)
        self.print("Writing video in color", out_video_fn)
        write_p1_video(self.video, labels, period, out_video_fn,
                       interval=P1_FRAME_INTERVAL, repeat=4, loopable_grid=self.is_loopable)

    def write_p2_video(self, final_vid_len, labels_ph2):
        M, N = self.M, self.N
        # Save as a video (or a few)
        per = p2_period_shortname(self.periods)
        # per += "labels-i{}".format(P2_FRAME_INTERVAL)
        if self.NO_STATIC: per += "-no-static"
        else: per += "_static{}".format(self.static_cost)
        out_video_fn = helpers.get_loop_fn(self.args, period=per, theta=self.theta)
        if final_vid_len > LONGEST_P2_VID:
            print("Final video too long (%d frames) - trimming to %d frames" % (final_vid_len, LONGEST_P2_VID))
            final_vid_len = LONGEST_P2_VID
        self.print("Writing final loop video to:", out_video_fn, "length:", final_vid_len)
        # outVid = np.empty([final_vid_len, M, N, 3])
        tm_cp_name = helpers.get_checkpoint_name("time_map", self.args, per, self.theta)
        loaded_tm = False
        try:
            time_map = np.load(tm_cp_name)
            loaded_tm = True
            print("Loaded time map")
        except (IOError, FileNotFoundError) as e:
            print("no time map to load")
            time_map = np.zeros( (final_vid_len, M, N), dtype="uint8" )
        if not loaded_tm:
            for t in tqdm(range(final_vid_len), disable=self.disable_tqdm):
                nodeid = 0
                for i in range(M):
                    for j in range(N):
                        tp = self.gc_lbl_to_tp(nodeid, labels_ph2[nodeid])
                        s, p = tp
                        time = s
                        if p > 1: time += (t - s) % p

                        val = self.V_color(cCoord(i, j), time)
                        # outVid[t, i, j] = val
                        time_map[t, i, j] = time

                        nodeid += 1

            print("Saving time map to", tm_cp_name)
            np.save(tm_cp_name, time_map)
        # skvideo.io.vwrite(out_video_fn, outVid)
        # self.print("Wrote video", out_video_fn)

        if self.args.upscaled_video:
            out_video_fn = helpers.get_loop_fn(self.args, period=per, theta=self.theta, notes='upscaled')
            print("Computing and saving higher-resolution video", out_video_fn)
            hi_res = skvideo.io.vread(self.args.upscaled_video)
            _, hM, hN, _ = hi_res.shape
            dM, dN = hM // M, hN // N
            # hi_res_time_map = np.kron(time_map, np.ones((dM, dN)))
            # write_video(hi_res, hi_res_time_map, out_video_fn, final_vid_len)
            write_HD_P2(hi_res, time_map, (dM, dN), out_video_fn, final_vid_len)
            print("wrote video")


    def distance(self, x, xframe, z, zframe):
        theta = self.theta
        a = self.V_color(x, xframe)
        b = self.V_color(z, zframe)
        emb = 0
        rgb = np.sum((b - a)**2)
        assert rgb >= 0

        if self.USE_EMBEDDINGS and theta > 0:
            emb = embeddings_helper.distance(x, xframe, z, zframe)
            assert emb >= 0
            ret = theta * emb + (1 - theta) * rgb
        else:
            ret = rgb

        # don't want ret to be around that 441.86, since we have no completely black or white pixels right now
        # assert not (ret < 442 and ret > 441)
        assert ret >= 0

        ret *= DISTANCE_SCALE

        self.tmp_dist_ct += 1
        if self.tmp_dist_ct % 50000 == 0 and self.logFile is not None:
            self.logFile.write("{:.4f}\t{:.4f}\t{:.4f}\n".format(rgb, emb, ret))

        return ret

    def gamma(self, x, z):
        # TODO: we could speed this up, I'm sure. Don't need to calculate all 'T' values each time
        T = self.T
        changes = np.empty(T)
        for t in range(T):
            changes[t] = np.linalg.norm(self.V_color(x, t) - self.V_color(z, t))
        mad = robust.mad(changes)
        return 1 / (1 + (LAMBDA_S * mad))

    def psi(self, x, z, tp_x, tp_z):
        T = self.T
        distance = self.distance
        dist_sum = 0.0
        if is_static(tp_x) and is_static(tp_z):
            # Both pixels are static
            ret = distance(x, tp_x.start, x, tp_z.start) + \
                  distance(z, tp_x.start, z, tp_x.start)
            return ret
        elif is_static(tp_x):
            phi_z = gen_phi(tp_z)
            for t in range(T - 1):
                dist_sum += distance(x, tp_x.start, x, phi_z(t)) + \
                            distance(z, tp_x.start, z, phi_z(t))
            ret = (1 / T) * dist_sum
            # ret2 = lookup_table.calc_energy_case2(x, z, tp_x, tp_z, self.video)
            # assert ret == ret2
            return ret
        elif is_static(tp_z):
            phi_x = gen_phi(tp_x)
            for t in range(T - 1):
                try:
                    dist_sum += distance(x, tp_z.start, x, phi_x(t)) + \
                                distance(z, tp_z.start, z, phi_x(t))
                except Exception as e:
                    print("-"*5, "Case4 hit an exception in distance()", "-"*5)
                    traceback.print_stack()
                    print(e)
                    print(x, z, tp_x, tp_z, phi_x(t))
                    print("Quitting....")
                    exit(1)
            ret = (1 / T) * dist_sum
            # ret2 = lookup_table.calc_energy_case2(x, z, tp_x, tp_z, self.video)
            # assert ret == ret2
            return ret
        elif same_period(tp_x, tp_z):
            phi_x = gen_phi(tp_x)
            phi_z = gen_phi(tp_z)
            for t in range(0, tp_x.period - 1):
                dist_sum += distance(x, phi_x(t), x, phi_z(t)) + \
                            distance(z, phi_x(t), z, phi_z(t))
            ret = dist_sum / tp_x.period
            return ret
        else:
            # TODO - does this need to be rewritten? And does it include the error in the paper? (See Errata on Hoppe's page)
            lcm_T = int_lcm(tp_x.period, tp_z.period)
            phi_x = gen_phi(tp_x)
            phi_z = gen_phi(tp_z)
            # I think that this is right, but the paper is not explicitly clear
            _start = sys_time.perf_counter()
            for t in range(0, lcm_T - 1):
                dist_sum += distance(x, phi_x(t), x, phi_z(t)) + \
                            distance(z, phi_x(t), z, phi_z(t))
            if lcm_T is 0:
                self.print("WARNING! LCM was 0!!")
                self.print(x, z, tp_x, tp_z)
            ret = dist_sum / lcm_T
            # ret2 = lookup_table.calc_energy_case4(x, z, tp_x, tp_z)
            # assert ret == ret2
            assert ret >= 0
            return ret

    def unary_cost(self, loc, lbl):
        V_color, distance = self.V_color, self.distance
        T, theta = self.T, self.theta
        s = lbl.start
        p = lbl.period

        if not self.is_loopable[loc.i, loc.j]:
            return 10

        if is_static(lbl):
            return self.static_cost

        d_rgb = np.linalg.norm(V_color(loc, s) - V_color(loc, s + p)) ** 2 + \
                np.linalg.norm(V_color(loc, s - 1) - V_color(loc, s + p - 1)) ** 2
        discon = d_rgb
        if self.USE_EMBEDDINGS:
            d_emb = embeddings_helper.distance(loc, s, loc, s + p) ** 2 + \
                    embeddings_helper.distance(loc, s - 1, loc, s + p - 1) ** 2
            discon = (theta * d_emb) + ((1 - theta) * d_rgb)
        succ_var = np.empty(T - 1, dtype='float')
        for t in range(T - 1):
            # rgb = np.linalg.norm(V_color(loc, t) - V_color(loc, t-1))
            # emb = embeddings_helper.distance(loc, t, loc, t-1)
            succ_var[t] = distance(loc, t, loc, t - 1)
        factor = 1 / (1 + LAMBDA_T * robust.mad(succ_var))
        ret = discon * factor
        assert ret >= 0
        return ret

    def gen_pairwise_cost_fn(self, current_period):
        _period = current_period

        def pairwise_cost(node_x, node_z, p1_lbl_x, p1_lbl_z):
            """
            Calculate the pairwise cost/smoothness/spacial consistency term for a given period
            :param node_x:
            :param node_z:
            :param p1_lbl_x:
            :param p1_lbl_z:
            :return:
            """
            if self.beta == 0: return 0
            tp_x = self.p1_label_to_tp(p1_lbl_x, _period)
            tp_z = self.p1_label_to_tp(p1_lbl_z, _period)
            x = self.id_to_loc(node_x)
            z = self.id_to_loc(node_z)

            psi_val = self.psi(x, z, tp_x, tp_z)
            gam_val = self.gamma(x, z)
            ret = psi_val * gam_val * self.beta
            assert ret >= 0
            return ret

        return pairwise_cost

    def get_unary_cost_map(self, nnodes, nlabels, period):
        unary = np.empty((nnodes, nlabels))
        for node in tqdm(range(nnodes), disable=self.disable_tqdm):
            loc = self.id_to_loc(node)
            # for s in range(period):
            for lbl_id in range(nlabels):
                lbl = self.p1_label_to_tp(lbl_id, period)
                s = lbl_id * P1_FRAME_INTERVAL
                lbl2 = cTPair(s, period)
                assert lbl.start == lbl2.start

                unary[node, lbl_id] = self.unary_cost(loc, lbl)
        self.print("unary cost map min, max, dtype\t", np.min(unary), np.max(unary), unary.dtype)
        return unary

    def expansion(self, num_labels, unary_cost_map, pairwise_cost_fn):
        """
        Run the alpha expansion for a given graph cut problem. Reused for phases 1 and 2
        :param num_labels:
        :param unary_cost_map:
        :param pairwise_cost_fn:
        :return:
        """
        M = self.M
        N = self.N

        gc = self.initialize_graph(M, N, num_labels)
        gc.set_data_cost(unary_cost_map)

        # initialize graph cut with argmins from unary cost map
        min_labels = np.argmin(unary_cost_map, axis=1)
        try:
            for nodeid in range(min_labels.size):
                gc.init_label_at_site(nodeid, min_labels[nodeid])
        except Exception as e:
            print("unable to set initial labels. ahh well!")
            print(e)

        if self.do_log:
            self.logFile = open(helpers.get_log_fn("dist-smooth.tsv", self.theta, self.args), "w")
            self.logFile.write("RGB\tEmbedding\n")
        gc.set_smooth_cost_function(pairwise_cost_fn)

        # Run a-b expansion
        self.print("Running alpha-beta expansion with", num_labels, "labels. Started at", datetime.datetime.now())
        gc.expansion(niters=1)
        self.print("Finished at %s. ENERGY: %d (data: %d; smooth: %d)." %
              (datetime.datetime.now(), gc.compute_energy(), gc.compute_data_energy(), gc.compute_smooth_energy()))
        if self.logFile is not None:
            self.logFile.close()
            self.logFile = None

        return gc.get_labels()

    # PHASE 1 RUNNERS

    def phase_i_expansion(self, period):
        M = self.M
        N = self.N
        # We can start as late as `T-period`, since starting any later would run past the end of the video
        nlabels = (self.T - period) // P1_FRAME_INTERVAL

        # handle logging
        if self.do_log:
            self.logFile = open(helpers.get_log_fn("distances-unary.tsv", self.theta, self.args), "w")
            self.logFile.write("RGB\tEmbedding\n")

        was_USE_EMBEDDINGS = self.USE_EMBEDDINGS
        if self.args.no_temporal_embeddings:
            self.USE_EMBEDDINGS = False
        self.print("Generating and setting unary costs.", nlabels, "labels per node. Embeddings?", self.USE_EMBEDDINGS)
        unary_cost_map = self.get_unary_cost_map(M * N, nlabels, period)
        self.USE_EMBEDDINGS = was_USE_EMBEDDINGS

        # Wrap up logging
        if self.logFile is not None:
            self.logFile.close()
            self.logFile = None

        return self.expansion(nlabels, unary_cost_map, self.gen_pairwise_cost_fn(period))

    def calculate_or_load_phase_I(self, period):
        checkpoint_name = helpers.get_checkpoint_name("labels", self.args, period, theta=self.theta)

        labels = try_get_checkpoints(checkpoint_name, self.args.no_cache)
        checkpoint_exists = labels is not None
        if not checkpoint_exists:
            start = sys_time.time()
            labels = self.phase_i_expansion(period)
            self.print("phase 1 for this period took %.2f seconds" % (sys_time.time() - start))

            # Store it as a numpy object for next time
            if not self.args.no_cache:
                np.save(checkpoint_name, labels)
                self.print("Saved checkpoint", checkpoint_name)

            # self.save_label_images(labels, period)
            self.save_p1_video(labels, period)
        return labels

    def DO_PHASE_I(self, period_idx):
        """
        This is the "public" function that another piece of code might call. If running serially, this function will
        be called multiple times in serial, once per period (and therefore per Runner instance).
        If running with MPI, each process will call this once.
        :param period_idx:
        :return: The result of the expansion for a given period in phase 1
        """
        period = self.periods[period_idx]
        labels = self.calculate_or_load_phase_I(period)
        if not self.is_mpi:
            self.labels_ph1[period_idx] = labels
        return labels

    # PHASE 2 STUFF

    def gc_lbl_to_tp(self, nodeid, label):
        T = self.T
        T_mod = self.T // P2_FRAME_INTERVAL
        periods = self.periods
        labels_ph1 = self.labels_ph1
        if self.args.no_static:
            # There are no static pixels, so we instead grab right from a P1 result.
            _s = labels_ph1[label][nodeid] * P1_FRAME_INTERVAL
            _p = periods[label]
        elif label < T_mod:
            # The first 'T/p2interval' values correspond (not directly) to starting frames for static pixels
            # That is, these are static pixels, and we only consider T_mod possible frames
            _s = label * P2_FRAME_INTERVAL
            _p = 1
        elif label < T_mod + len(periods):
            # Else, we subtract 'T' and use that new value as an index into one of the previous phases
            _s = labels_ph1[label - T_mod][nodeid] * P1_FRAME_INTERVAL
            _p = periods[label - T_mod]
        else:
            raise Exception("Label {} is too big!".format(label))
        return cTPair(_s, _p)

    def get_unary_cost_map_ph2(self, nnodes, num_labels):
        unary = np.empty((nnodes, num_labels))
        for node in tqdm(range(nnodes), disable=self.disable_tqdm):
            loc = self.id_to_loc(node)
            for t in range(num_labels):
                tp = self.gc_lbl_to_tp(node, t)
                unary[node, t] = self.unary_cost(loc, tp)
        return unary

    def pairwise_cost_p2(self, node_x, node_z, gc_lbl_x, gc_lbl_z):
        gc_lbl_to_tp = self.gc_lbl_to_tp
        id_to_loc = self.id_to_loc
        psi, gamma = self.psi, self.gamma
        tp_x = gc_lbl_to_tp(node_x, gc_lbl_x)
        tp_z = gc_lbl_to_tp(node_z, gc_lbl_z)
        x = id_to_loc(node_x)
        z = id_to_loc(node_z)

        psi_val = psi(x, z, tp_x, tp_z)
        gam_val = gamma(x, z)
        ret = psi_val * gam_val * self.beta
        assert ret >= 0
        return ret

    def phase_ii_expansion(self, num_labels):
        was_USE_EMBEDDINGS = self.USE_EMBEDDINGS
        if self.args.no_temporal_embeddings:
            self.USE_EMBEDDINGS = False
        unary_cost_map = self.get_unary_cost_map_ph2(self.M * self.N, num_labels)
        self.USE_EMBEDDINGS = was_USE_EMBEDDINGS
        smooth_fn = self.pairwise_cost_p2

        return self.expansion(num_labels, unary_cost_map, smooth_fn)

    def DO_PHASE_II(self):
        """
        The 'public' function that other code will call to do the final graph cut. This should only ever be called once
        per experiment (by the root node if running with MPI).
        :return: The labels for the final video
        """
        if self.args.skip_phase_ii:
            self.print("Skipping phase II")
            return None
        self.print("")
        self.print("-" * 10, "Starting phase two...", "-" * 10)
        self.print("")
        periods, args, T = self.periods, self.args, self.T
        # Make sure we consider static pixels in this part ==> T + len(periods)
        num_labels = len(periods)
        if not args.no_static:
            num_labels += T // P2_FRAME_INTERVAL
        # num_labels = len(periods) if args.no_static else T//P1_FRAME_INTERVAL + len(periods)
        print("phase 2 will have %d labels. There are %d periods and %d static frames" % (num_labels, len(periods), T//P2_FRAME_INTERVAL))
        print("video is %d frames long" % self.video.shape[0])

        prefix = "labels-i{}".format(P2_FRAME_INTERVAL)
        if args.no_static: prefix += "_no-static"
        else: prefix += "_static{}".format(self.static_cost)
        per = p2_period_shortname(self.periods)
        ckpt_name = helpers.get_checkpoint_name(prefix, args, period=per, theta=self.theta)

        labels = try_get_checkpoints(ckpt_name)
        if labels is None:
            start = sys_time.time()
            labels = self.phase_ii_expansion(num_labels)
            self.print("phase 2 took %.2f seconds" % (sys_time.time() - start))

            # Store it as a numpy object for next time
            if not args.no_cache:
                np.save(ckpt_name, labels)
                self.print("saved checkpoint", ckpt_name)

        lcm = np.lcm.reduce(periods)

        # self.save_label_images(labels, "all")
        self.write_p2_video(lcm, labels)
        return labels


