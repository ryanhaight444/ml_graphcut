import numpy as np
from scipy.ndimage.filters import gaussian_filter

EPSILON = 10.0
L_UNCHANGING = 1
L_UNLOOPABLE = 2
L_LOOPABLE   = 3


def get_mask(V):
    return is_loopable(classify_pixels(V))


def is_loopable(labels):
    """
    Determine if a pixel is loopable or not (convert labels (3 types) to booleans)
    :param labels: The labels generated from classify_pixels()
    :return: An M*N array of booleans where each loopable pixel has the value of True
    """
    ret = np.zeros(labels.shape, dtype="bool")
    ret[labels == L_LOOPABLE] = True

    # TEMP
    n_loopable = np.sum(ret.astype("int"))
    n_possible = labels.size
    percent = float(n_loopable) / n_possible
    print("There are %d (out of %d) loopable pixels (%0.2f%%)" % (n_loopable, n_possible, percent*100))
    # END TEMP

    return ret


def classify_pixels(V):
    """
    For every pixel in the input video, determine if it is able to be looped.
    :param V: The input video (RGB values 0..255)
    :return:  An M*N array of ints where each value represents one of {unchanging, unloopable, loopable}
    """
    # Make sure our video is in the range [0...255] and isn't all one solid color
    vid_min, vid_max = np.min(V), np.max(V)
    assert vid_min >= 0
    assert vid_max - vid_min > 0

    T, M, N, n_ch = V.shape
    rises = np.zeros((M, N, n_ch), dtype="bool")
    falls = np.zeros((M, N, n_ch), dtype="bool")

    for ch in range(n_ch):
        # For each color channel, we determine if each pixel "rises" (goes from low to high intensity)
        # and/or "falls" (high to low).
        lo = np.ones((M, N)) * 255
        hi = np.ones((M, N))
        for t in range(T):
            _V = V[t, ..., ch]
            lo = np.minimum(lo, _V)
            hi = np.maximum(hi, _V)

            rises[_V - lo > EPSILON, ch] = True
            falls[hi - _V > EPSILON, ch] = True

    # Based on the method in the paper, we now calculate which pixels (for each color channel) have the three
    # possible labels.
    unchanging = np.ones((M, N), dtype="bool")
    unloopable = np.zeros((M, N), dtype="bool")
    for ch in range(n_ch):
        _change = np.logical_not(rises[..., ch]) & np.logical_not(falls[..., ch])
        unchanging = unchanging & _change

        _loop = np.logical_xor(rises[..., ch], falls[..., ch])
        unloopable = np.logical_or(unloopable, _loop)

    loopable = ~np.logical_or(unchanging, unloopable)

    # Now, we gaussian blur all three. They may now overlap some, so the order in which we
    # assign the final labels matters. We assign 'loopable' last because we want as many
    # pixels to be loopable as possible
    unchanging = np.rint(gaussian_filter(unchanging*255, 7, order=0) / 255).astype("bool")
    unloopable = np.rint(gaussian_filter(unloopable*255, 7, order=0) / 255).astype("bool")
    loopable = np.rint(gaussian_filter(loopable * 255, 7, order=0) / 255).astype("bool")

    labels = np.zeros((M, N), dtype="int")
    labels[unchanging] = L_UNCHANGING
    labels[unloopable] = L_UNLOOPABLE
    labels[loopable]   = L_LOOPABLE
    return labels

if __name__ == "__main__":
    import skvideo.io
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser(description='Visualize the looping mask')
    parser.add_argument("source",
                        help="The source video")
    parser.add_argument("--save", action="store_true",
                        help="Store the mask as a numpy file")
    args = parser.parse_args()
    vid_fn = args.source

    print("Loading video...")
    vid = skvideo.io.vread(vid_fn)
    print("Computing labels...")
    classes = classify_pixels(vid)
    print("Displaying labels...")
    plt.imshow(classes)
    plt.show()

    # Show loopable pixels
    loopable_pixels = is_loopable(classes)
    plt.imshow(loopable_pixels.astype("int"))
    plt.show()

    # Overlay
    overlay = vid[0]
    overlay[..., 0] = 0
    overlay[loopable_pixels, ..., 0] = 255
    plt.imshow(overlay)
    plt.show()

    if args.save:
        np.save("is_loopable_" + vid_fn[vid_fn.rfind("/")+1:], loopable_pixels)
