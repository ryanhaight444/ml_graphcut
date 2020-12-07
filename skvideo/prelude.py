import skvideo.io
import skvideo.utils
from collections import namedtuple
import argparse
import datetime

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
parser.add_argument("video",
                    help="Which video file to process")
parser.add_argument("-e", "--embeddings",
                    help="Name of the embeddings (e.g. '128_projected')")
parser.add_argument("--upscaled-video",
                    help="Path to a higher-res version of the video for output")
parser.add_argument('--small-size', action='store_true',
                    help='DEPRECATED. Was used to run computation on a small part of the video')
parser.add_argument('--small-time', action='store_true',
                    help='DEPRECATED. Use "--num-frames" instead. Only run computation on a few frames of the video')
parser.add_argument('--num-frames', type=int, default=0,
                    help='Manually set the number of frames (including embeddings) to load')
parser.add_argument('-o', '--img-file',
                    help="Where to store the output image")
parser.add_argument("-B", "--beta", type=int, default=BETA_DEFAULT,
                    help="Adjust the multiplier for the smoothness cost")
parser.add_argument("--theta", type=float, default=[0.5], nargs="*",
                    help="The relative weight of the embeddings over RGB values. Range [0, 1]")
parser.add_argument("-P", "--periods", type=int, nargs="+", default=-1,
                    help="Manually pick the periods to try")
parser.add_argument("--fps60", action="store_true",
                    help="Use if videos were 60 FPS instead of 30. Ignore every other frame of vid and embedding")
parser.add_argument("--no-static", action="store_true",
                    help="If included, don't include static pixels in Phase II")
parser.add_argument("-t", "--temp", "--tmp", action="store_true",
                    help="Store files in root dir (temp for testing)")
parser.add_argument("--tag",
                    help="Include this string in output file names")
parser.add_argument("--no-cache", action="store_true",
                    help="Do not read or write cache/checkpoint files")
parser.add_argument("--do-log", action="store_true",
                    help="Store logs about the program, whatever that may mean at the time")
parser.add_argument("--lda-frame-0-broken", action="store_true",
                    help="Some LDA[..., 0] embeddings are broken. Zero them out")
parser.add_argument("--skip-phase-ii", action="store_true")
parser.add_argument("--no-temporal-embeddings", "--nte", action="store_true",
                    help="Skip embeddings when calculating the temporal (unary) cost")
parser.add_argument("--static-cost", type=int, default=10,
                    help="The cost of setting a pixel to be static")
args = parser.parse_args()

if args.small_size:
    raise Exception("--small-size argument is no longer supported")

# RUNTIME VARIABLES
beta = args.beta
theta = args.theta[0] if len(args.theta) else 0.5
if theta < 0 or theta > 1: theta = 0.5
filename = args.video
args.input_file = filename
USE_EMBEDDINGS = True if args.embeddings else False
NO_STATIC = args.no_static

print(datetime.datetime.now(), "Starting up. Loading video")

num_frames = args.num_frames
vid_color = skvideo.io.vread(filename, num_frames=num_frames)
vid_color = vid_color.astype("float32")
if args.fps60:
    vid_color = vid_color[::2]
T, M, N, _ = vid_color.shape

periods = args.periods if args.periods is not -1 else range(30, T-4, 5)
if args.periods == -1:
    print("No period(s) provided, so defaulting to:", list(periods), "(", len(periods), "experiments)")

assert periods[-1] < T
