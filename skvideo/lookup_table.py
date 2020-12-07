import numpy as np

_table = np.empty((1,1))
_table2 = np.empty((1,))

def compute_lookup_table(source):
    """
    Create the cumulative sum lookup table for a given set of values
    :param source: either the embeddings or the RGB video
    :return: the cumulative sum table
    """
    global _table
    global _table2

    table = np.cumsum(source, axis=0)
    table2 = np.cumsum(source ** 2, axis=0)

    _table = table
    _table2 = table2

    return table, table2


def cum_sum(loc, lb, ub, table):
    """
    Calculate the cumulative sum based on the precomputed table
    :param loc:     The (i, j) location of the pixel
    :param lb:      Lower bound of the summation
    :param ub:      Upper bound of the summation
    :param table:   The pre-computed lookup table
    :return: (number) The sum from 'lb' to 'ub' in the original source
    """
    assert lb <= ub, "cum_sum: lb {} was greater than ub {}!".format(lb, ub)
    lower = table[lb, loc.i, loc.j]
    upper = table[ub, loc.i, loc.j]
    return upper - lower


def case2_helper(loc, tp_a, tp_b, source, table):
    """
    Implements most of the nuts and bolts for one of the calculations in Case 2.
    :param loc:    The pixel coordinates
    :param tp_a:   (start, period) for this pixel
    :param tp_b:   (start, period) for adjacent pixel
    :param source: The original values (eg embeddings or RGB video)
    :param table:  The pre-computed lookup table
    :return: (number) One half of the Phi calculation for Case 2.
    """
    sx = tp_a.start
    sz = tp_b.start
    pz = tp_b.period
    a = source[sx, loc.i, loc.j] ** 2
    b = (2.0 * source[sx, loc.i, loc.j]) / pz
    c = cum_sum(loc, sz, sz + pz - 1, table)
    d = (1.0 / pz) * cum_sum(loc, sz, sz + pz - 1, _table2)

    # Sometimes, a < b*c + d, so the return value would be negative. This seems to occur naturally, and there's no
    # mention of it in the paper. It's unclear to me what might be needed to "fix" this issue, so instead I just return
    # the max of zero and the calculated result (ie I return zero if I was going to return a negative number).

    # vector - ( vector <pointwise multiply> vector ) + vector --> scalar
    ret = np.sum(a - (b * c) + d)
    if ret < 0:
        # print(sx, sz, pz, "\t", a, (b*c) + d, ret, np.linalg.norm(a - (b*c) + d))
        ret = 0
    return ret


def calc_energy_case2(x, z, tp_x, tp_z, source):
    """
    Calculate the energy value (Phi) with the lookup table. This is the second case in the Spacial Consistency section
    of the video looping paper.
    :param x:      The location of pixel x
    :param z:      The location of pixel z
    :param tp_x:   The start frame and period length for pixel x
    :param tp_z:   The start frame and period length for pixel z
    :param source: The original values (eg embeddings or RGB video)
    :return: (number) The energy based on the function in the paper
    """
    table = _table
    sum_a = case2_helper(x, tp_x, tp_z, source, table)
    sum_b = case2_helper(z, tp_x, tp_z, source, table)
    if sum_a < 0 or sum_b < 0:
        print(x, z, tp_x, tp_z, sum_a, sum_b)
    assert sum_a >= 0, "calc_energy_case2: sum_a < 0"
    assert sum_b >= 0, "calc_energy_case2: sum_b < 0"
    return sum_a + sum_b


def case4_helper(loc, tp_x, tp_z, table, table2):
    sx, px = tp_x.start, tp_x.period
    sz, pz = tp_z.start, tp_z.period
    m = tp_x.period
    n = tp_z.period
    channels = table.shape[-1]
    # if phi_x(0) <= phi_x(m-1) or phi_z(0) <= phi_z(n-1) or phi_x(0) <= phi_x(m-1) or phi_z(0) <= phi_z(n-1):
    #     print(loc, tp_x, tp_z, m, n, "---\t", phi_x(0), phi_x(m-1), phi_z(0), phi_z(n-1))
    # assert phi_x(0) <= phi_x(m-1), "case4_helper: phi_x(0) > phi_x(m-1)\t{}, {}".format(phi_x(0), phi_x(m-1))
    # assert phi_z(0) <= phi_z(n-1), "case4_helper: phi_z(0) > phi_z(n-1)\t{}, {}".format(phi_z(0), phi_z(n-1))
    # assert phi_x(0) <= phi_x(m-1), "case4_helper: phi_x(0) > phi_x(m-1)\t{}, {}".format(phi_x(0), phi_x(m-1))
    # assert phi_z(0) <= phi_z(n-1), "case4_helper: phi_z(0) > phi_z(n-1)\t{}, {}".format(phi_z(0), phi_z(n-1))
    # print("case4_helper: calc a2")

    # The time mapping isn't working quite how I expect, but I think the idea is that we sum from the first time
    # to the last time of a given loop for this pixel. I think that the phi() mapping in the paper theoretically works,
    # but with the way we use the lookup table we need to give it the exact extreme ends. Technically, looping over
    #   t from phi(0) to phi(period-1)
    # should give every single frame in a pixel's looping period. However, because the start frame may not be frame 0,
    # and phi() does some modulo stuff to make it  all work out, that means that phi(0) and phi(period-1) may be
    # "random" frames within the possible range. Instead, since we know the start time and period for each pixel,
    # we just use that instead.

    # a2 = (1.0 / m) * cum_sum(loc, phi_x(0), phi_x(m-1), table2)
    a2 = (1.0 / m) * cum_sum(loc, sx, sx+px, table2)

    # b2 = (1.0 / n) * cum_sum(loc, phi_z(0), phi_z(n-1), table2)
    b2 = (1.0 / n) * cum_sum(loc, sz, sz+pz, table2)

    ab_a = 2.0 / (m*n)
    ab_b = cum_sum(loc, sx, sx + px, table)
    ab_c = cum_sum(loc, sz, sz + pz, table)
    # ab = (2.0 / (m * n)) * cum_sum(loc, sx, sx + px, table) * cum_sum(loc, sz, sz + pz, table)
    ab = ab_a * ab_b * ab_c
    # print(ab_a, ab_b, ab_c, ab)
    #
    # print(a2, b2, ab)
    # print("a2 + b2 - ab")
    # print(a2 + b2 - ab)

    return np.linalg.norm(a2 + b2 - ab) / channels
    # return np.linalg.norm(a2 + b2) - ab


def calc_energy_case4(x, z, tp_x, tp_z):
    print("----------------> case 4")
    table = _table
    table2 = _table2
    for_x = case4_helper(x, tp_x, tp_z, table, table2)
    for_z = case4_helper(z, tp_x, tp_z, table, table2)
    # print("case 4 calculated:", for_x, for_z)
    assert np.isscalar(for_x), "case4: for_x is not a scalar"
    assert np.isscalar(for_z), "case4: for_z is not a scalar"
    assert for_x >= 0, "case4: for_x < 0"
    assert for_z >= 0, "case4: for_z < 0"
    return for_x + for_z
