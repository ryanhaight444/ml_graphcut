from mpi4py import MPI
from prelude import *
from runner import Runner

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

# Now, go through and do Phase I for each period individually based on the process's rank
for theta_index, theta_val in enumerate(args.theta):
    if theta_index % nproc == rank:
        runner = Runner(vid_color, periods, args, is_mpi=True, custom_print_fn=rprint, theta=theta_val)
        for period_index, period in enumerate(args.periods):
            rprint("Running for theta=%.2f (index %d), period=%d" % (theta_val, theta_index, period))
            runner.DO_PHASE_I(period_index)
            rprint("Finished theta=%.2f (index %d). Freeing memory" % (theta_val, theta_index))

rprint("process exiting")
