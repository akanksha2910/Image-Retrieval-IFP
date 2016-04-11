#!/usr/bin/env sh
#Create the siamese alisc lmdb inputs
# N.B. set the path to the siamese alisc train + val data dirs

DB=data/alisc/
LIST_ROOT=data/alisc/list/
TOOLS=build/tools

TRAIN_DATA_ROOT=data/alisc/final_eva_image/

# Set RESIZE=true to resize the images to 224x224. Leave as false if images have
# already been resized using another tool.
RESIZE=true
ENCODED=true
if $RESIZE; then
  RESIZE_HEIGHT=146
  RESIZE_WIDTH=146
  SHUFFLE=false
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  exit 1
fi


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=$SHUFFLE \
    --encoded=$ENCODED \
    $TRAIN_DATA_ROOT \
    $LIST_ROOT/list_final_eva_image_label.txt\
    $DB/imgeval_all_146_lmdb
echo "Done."

