#!/usr/bin/env sh
#Create the siamese alisc lmdb inputs
# N.B. set the path to the siamese alisc train + val data dirs

DB=data/alisc/
LIST_ROOT=data/alisc/list/
TOOLS=build/tools

TRAIN_DATA_ROOT=data/alisc/query_image/

# Set RESIZE=true to resize the images to 224x224. Leave as false if images have
# already been resized using another tool.
RESIZE=true
ENCODED=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
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
    $LIST_ROOT/valid_image_label.txt\
    $DB/valid_image_label_256_lmdb
echo "Done."

