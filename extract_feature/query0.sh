TOOLS=./build/tools
FEATURES=data/alisc/gfea100R37/query
FEATURESNAME=pool5/max_7x7_s1_128
FEATURESFNAME=pool5_max_7x7_s1_128
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME

echo "extracting features query"
$TOOLS/extract_features models/a128googlenet/finetune100R37k/a128_googlenet_quick_iter_12000.caffemodel models/a128googlenet/deploy160/query.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/npy 41 lmdb GPU 1

echo " done query"
