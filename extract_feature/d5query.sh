TOOLS=./build/tools
FEATURES=data/alisc/gfea240k/query
FEATURESNAME=inception_5b/output
FEATURESFNAME=inception_5b_output
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME

echo "extracting features query"
$TOOLS/extract_features models/a128googlenet/finetune/a128_googlenet_quick_iter_240000.caffemodel models/a128googlenet/lmdb_deploy/query.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/npy 41 lmdb GPU 1

echo " done query"
