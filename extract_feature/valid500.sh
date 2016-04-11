TOOLS=./build/tools
#FEATURESG=data/alisc/gfea170
FEATURES=data/alisc/gfea170k/valid
FEATURESNAME=pool5/7x7_s1_128
FEATURESFNAME=pool5_7x7_s1_128
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME

echo "extracting features valid"
$TOOLS/extract_features models/a128googlenet/finetune/a128_googlenet_quick_iter_170000.caffemodel models/a128googlenet/lmdb_deploy/valid500.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/npy500 5 lmdb GPU 1

echo " done valid"
