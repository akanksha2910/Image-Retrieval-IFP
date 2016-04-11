TOOLS=./build/tools
FEATURESG=data/alisc/gfea170kall
FEATURES=data/alisc/gfea170kall/eval
FEATURESNAME=pool5/7x7_s1_128
FEATURESFNAME=pool5_7x7_s1_128
mkdir $FEATURESG
mkdir $FEATURES
#mkdir $FEATURES/$FEATURESFNAME

echo "extracting featrue eval"

$TOOLS/extract_features models/a128googlenet/finetune/a128_googlenet_quick_iter_170000.caffemodel models/a128googlenet/lmdb_deploy/eval.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME 3195334 lmdb GPU 1

echo " done 3195334 eval"
