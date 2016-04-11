TOOLS=./build/tools
#FEATURESG=data/alisc/gfea170
FEATURES=data/alisc/gfea240k/valid
FEATURESNAME=inception_5b/output
FEATURESFNAME=inception_5b_output
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME

echo "extracting features valid"
$TOOLS/extract_features models/a128googlenet/finetune/a128_googlenet_quick_iter_240000.caffemodel models/a128googlenet/lmdb_deploy/valid.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/npy 13 lmdb GPU 1

echo " done valid"
