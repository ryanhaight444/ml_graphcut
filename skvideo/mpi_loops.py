from mpi4py import MPI
import numpy as np
from prelude import *
from runner import Runner
from datetime import datetime

# MPI Variables
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

def rprint(*a, **kwargs):
    """
    Regular print, but it puts your process rank in front.
    (Definitely possible to have some race conditions here if you print too much from different processes, though,
    so just don't go crazy with this function and expect it to be perfect.)
    :param a:
    :param kwargs:
    :return:
    """
    print("[%d] " % rank, end="")
    print(*a, **kwargs)


#######################################################################################################################
##
## Time to get to business!
##
#######################################################################################################################


# Set up the variables we'll use
# TODO: should this be `[None] * len(periods)` so that we don't eat up a bunch of extra memory in each process just to
# store this empty numpy array? Also at the very least shouldn't it be empty, not zeros?
labels_ph1 = [np.zeros((M*N), dtype="int32")] * len(periods)

# Now, go through and do Phase I for each period individually based on the process's rank
for index, period in enumerate(periods):
    if index % nproc == rank:
        rprint("Running for period=%d (index %d)" % (period, index))
        runner = Runner(vid_color, periods, args, is_mpi=True, custom_print_fn=rprint, theta=theta)
        labels = runner.DO_PHASE_I(index)
        labels_ph1[index] = labels
        rprint("Finished period=%d (index %d). Freeing memory" % (period, index))
        runner = None # free up the memory

# Now, combine all those results on the root node
for index, period in enumerate(periods):
    owner = index % nproc
    if rank == 0:
        # ROOT process receives
        if owner != 0:
            comm.Recv(labels_ph1[index], source=owner)
    elif owner == rank:
        # Other processes send only their computations
        comm.Send(labels_ph1[index], dest=0)
        labels_ph1[index] = None

# now, we do phase 2
# Want to find the best period from all possible periods (`periods` array)
# The cost of selecting a given period is the cost of using that period and
#   the associated start time with that period at that pixel
if rank==0:
    runner = Runner(vid_color, periods, args, True, rprint, theta=theta)
    runner.provide_all_p1_labels(labels_ph1)
    runner.DO_PHASE_II()

rprint("process exiting at", datetime.now())
