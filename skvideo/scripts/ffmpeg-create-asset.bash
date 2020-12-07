#!/bin/bash

USAGE="
USAGE:\n
\t./ffmpeg-create-asset.bash <input path> <video common name> <new size (WxH)> <start time offset>\n
"

# Input parameters (arguments)
INPUT_PATH=$1
VIDEO_NAME=$2
NEW_SIZE=$3
START_TIME=$4

if [ $# -ne 4 ]; then
    echo -e $USAGE
    exit
fi

#                              no audio                      duration
#              compression         v         framerate           v      play on non-VLC
OUT_FLAGS="-c:v libx264 -crf 18   -an   -filter:v fps=fps=30   -t 5s   -pix_fmt yuv420p"
OUT_DIR="/projects/wehrresearch/ml_graphcut/assets"
OUT_NAME="${VIDEO_NAME}-${NEW_SIZE}.mp4"
CMD="ffmpeg -ss $START_TIME -i $INPUT_PATH -s $NEW_SIZE $OUT_FLAGS ${OUT_DIR}/${OUT_NAME}"

echo $CMD >> conversion_command_history
echo $CMD
$($CMD)
