TOOLS=./build/tools
FEATURES=data/alisc/features22/eval

$TOOLS/extract_features models/bvlc_googlenet/finetune/bvlc_googlenet_quick_iter_220000.caffemodel models/bvlc_googlenet/lmdb_deploy/eval_last1045801.prototxt pool5/7x7_s1 $FEATURES/pool5_7x7_s1_eval_last1045801 118 lmdb GPU 0
