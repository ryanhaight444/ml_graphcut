import skimage
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Adjust image contrast")
parser.add_argument("file", type=str, nargs=1,
                    help="The image to adjust")
args = parser.parse_args()

fn = args.file[0]

img = skimage.io.imread(fn)

largest = max(np.concatenate(img))

mult = 255 // largest

img = img * mult

skimage.io.imsave(fn, img)
