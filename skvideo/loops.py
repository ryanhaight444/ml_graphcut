from prelude import *
from runner import Runner

r = Runner(vid_color, periods, args, theta=theta)
for index, period in enumerate(periods):
    print("-"*15, "Starting period", period, "-"*15)
    r.DO_PHASE_I(index)

if len(periods) > 1:
    r.DO_PHASE_II()
else:
    print("Skipping Phase II since there was only 1 period")
