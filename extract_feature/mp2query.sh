TOOLS=./build/tools
FEATURESG=data/alisc/gfea3B1k
FEATURES=data/alisc/gfea3B1k/query
FEATURESNAME=inception_5b/output,pool5/7x7_s1_128
FEATURESFNAME1=inception_5b_output  
FEATURESFNAME2=pool5_7x7_s1_128
mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME1
mkdir $FEATURES/$FEATURESFNAME2

echo "extracting featrue query"
$TOOLS/extract_features models/a128googlenet/finetuneB3/a128_googlenet_quick_iter_1000.caffemodel models/a128googlenet/lmdb_deploy/query.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/npy,$FEATURES/$FEATURESFNAME2/npy 41 lmdb GPU 1
echo "done query"
