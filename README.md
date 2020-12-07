
# ml_graphcut

## Installing and Running Noah's Looping Stuff

First, clone the repo.

Then, run these commands

```bash
cd ml_graphcut/skvideo
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
cd ../pyGCO-pkg/
make
cd ../skvideo/
```

Next, edit `gco/cgco.py` in a text editor. Update the path in line 9 so that
it points at the `../pyGCO-pkg/` directory that we ran `make` in earlier.

Now, you can run the `loop.py` file with whatever video you want.
For example:

```bash
python loops.py /projects/wehrresearch/ml_graphcut/assets/brink2-320x180.mp4
```

You'll likely want to include some of the other parameters/arguments,
as the defaults may not be what you want.
For example, if you specify `-P` (`--periods`), you can tell it how many frames
you want each period to be for Phase I. So if you run with `-P 20`, it will
only calculate Phase I once with a period of 20.
Run `python loops.py --help` for more details on arguments, or see below.

## Recent copy of `usage` for `loops.py`

```
usage: loops.py [-h] [-e EMBEDDINGS] [--small-size] [--small-time]
                [--num-frames NUM_FRAMES] [-o IMG_FILE] [-B BETA]
                [--theta [THETA [THETA ...]]] [-P PERIODS [PERIODS ...]]
                [--no-static] [-t] [--tag TAG] [--no-cache] [--do-log]
                video

Graph cuts and such for loops

positional arguments:
  video                 Which video file to process

optional arguments:
  -h, --help            show this help message and exit
  -e EMBEDDINGS, --embeddings EMBEDDINGS
                        Name of the embeddings (e.g. '128_projected')
  --small-size          DEPRECATED. Was used to run computation on a small
                        part of the video
  --small-time          DEPRECATED. Use "--num-frames" instead. Only run
                        computation on a few frames of the video
  --num-frames NUM_FRAMES
                        Manually set the number of frames (including
                        embeddings) to load
  -o IMG_FILE, --img-file IMG_FILE
                        Where to store the output image
  -B BETA, --beta BETA  Adjust the multiplier for the smoothness cost
  --theta [THETA [THETA ...]]
                        The relative weight of the embeddings over RGB values.
                        Range [0, 1]
  -P PERIODS [PERIODS ...], --periods PERIODS [PERIODS ...]
                        Manually pick the periods to try
  --no-static           If included, don't include static pixels in Phase II
  -t, --temp, --tmp     Store files in root dir (temp for testing)
  --tag TAG             Include this string in output file names
  --no-cache            Do not read or write cache/checkpoint files
  --do-log              Store logs about the program, whatever that may mean
                        at the time
```

## Installing and Running Ryan's Embedding Stuff

First, clone the repo.

Then, run these commands

```bash
cd ml_graphcut/skvideo/gen_embeddings
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```
Inside the `/gen_embeddings`directory are two scripts (for convenience) that will drive the two main programs of interest- `dim_reduction.py` which you can run manually or with the `red_driver.sh` script. `gen_pixel_embeddings.py` can be ran manually or with the `gen_driver.sh` script.

`gen_pixel_embeddings.py` is currently configured to take in a video and output a frame of embeddings into a folder, specified by the user with the `--out_dir`argument, where in each frame of embeddings will be a output to a separate file (due to memory constraints) as a numpy array. 

`dim_reduction.py` is currently configured to perform a user specified dimensional reduction technique(`--mode`) down to a dimension specified by the user (`--new_size`). Each reduced frame will be output in the same same format as the original embeddings. In order to use`dim_reduction.py` ffmpeg must be installed, which can be done locally on the lab machines.

Other files of interest, are `vis.py` which can be used to transform a numpy array of semantic labels for an image into a labeled color map of an image. `auto_encoder.py` is a simple two layer auto encoder in pytorch that can be used for dimensionality reduction. 

Much of the code for generating the embeddings has been modified from the original, a suite of semantic segmentation models trained of ADE20k located on [github](https://github.com/CSAILVision/semantic-segmentation-pytorch). Currently only the default model (resnet50dilated + ppm_deepsup) is modified to output embeddings, however this can be changed by editing `models/models.py`.


