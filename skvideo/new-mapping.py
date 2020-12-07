import numpy as np
import skvideo.io
import skimage.io
from collections import namedtuple
import gco


def cPixel(x, y):
    return Pixel(x=x, y=y)
def cCoord(i, j):
    return Coord(i=i, j=j)

Pixel = namedtuple("Pixel", "x,y")
Label = namedtuple("Label", "start,period")
Coord = namedtuple("Coord", "i,j")

filename = "assets/marine_01-5s-300x200.mp4"
vid = skvideo.io.vread(filename,
                       outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
T, M, N = vid.shape
M, N = 8, 10
width = N

def id_to_loc(id):
    if id < 0: raise Exception("ID must be >= 0")
    key = id
    return cCoord(key // width, key % width)

def loc_to_id(loc):
    i, j = loc
    return i*width + j


out_img_c = np.empty((M,N,3), dtype="uint8")
out_img_g = np.empty((M,N),  dtype="float")
max_id = M * N
def get_color(id):
    RGBint = id / max_id
    # RGBint *= 255**3
    RGBint *= 255
    RGBint = int(RGBint)
    blue = RGBint & 255
    green = (RGBint >> 8) & 255
    red = (RGBint >> 16) & 255
    red = 255
    return [red, green, blue]


###############################################################################
# Unit Tests, of sorts, to make sure mapping works as intended
###############################################################################

assert id_to_loc(0).i == 0
assert id_to_loc(0).j == 0

assert id_to_loc(1).i == 0
assert id_to_loc(1).j == 1

true_id = 0
for i in range(M):
    for j in range(N):
        here = cCoord(i, j)
        got_id = loc_to_id(here)

        assert got_id == true_id

        got_loc = id_to_loc(true_id)
        assert got_loc.i == i
        assert got_loc.j == j

        out_img_c[i, j] = get_color(got_id)
        out_img_g[i, j] = got_id / max_id

        true_id += 1


gc = gco.GCO()
# M*N nodes in graph, and start frame from 0..period
gc.create_general_graph(M*N, 5)
for i in range(M):
    for j in range(N):
        here = cCoord(i, j)
        down = cCoord(i+1, j)
        rite = cCoord(i, j+1)
        idh = loc_to_id(here)
        idd = loc_to_id(down)
        idr = loc_to_id(rite)
        if i < M-1:
            # Add neighbor below us
            gc.set_neighbor_pair(idh, idd, 1)
        if j < N-1:
            # Add neighbor to the right of us
            gc.set_neighbor_pair(idh, idr, 1)



# Uncomment to make gradient images. Must be small size (~ 10x30) to see them well.
# skimage.io.imsave("labels-c.png", out_img_c)
# print("------------")
# skimage.io.imsave("labels-g.png", out_img_g)
