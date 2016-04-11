#!/usr/bin/env sh
#Create the siamese alisc lmdb inputs
# N.B. set the path to the siamese alisc train + val data dirs

DB=data/alisc/
LIST_ROOT=data/alisc/list/
TOOLS=build/tools

TRAIN_DATA_LMDB_ROOT=data/alisc/

# Set RESIZE=true to resize the images to 224x224. Leave as false if images have
# already been resized using another tool.
BACKEND=lmdb


if [ ! -d "$TRAIN_DATA_LMDB_ROOT" ]; then
  echo "Error: TRAIN_DATA_LMDB_ROOT is not a path to a directory: $TRAIN_DATA_LMDB_ROOT"
  exit 1
fi


echo "Creating train lmdb mean..."

GLOG_logtostderr=1 $TOOLS/compute_image_mean \
    --backend=$BACKEND \
    $TRAIN_DATA_LMDB_ROOT/imgtrainc10_256_lmdb \
    $DB/imgtrainc10_256_mean.binaryproto
echo "Done."

