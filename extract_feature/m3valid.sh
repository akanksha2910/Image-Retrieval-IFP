TOOLS=./build/tools
#FEATURESG=data/alisc/gf100f3ksd695
FEATURES=data/alisc/gf100f3ksd695/valid
FEATURESNAME=loss1/max_pool,loss2/max_pool,pool5/max_7x7_s1_128
FEATURESFNAME1=loss1_max_pool 
FEATURESFNAME2=loss2_max_pool
FEATURESFNAME3=pool5_max_7x7_s1_128
#mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME1
mkdir $FEATURES/$FEATURESFNAME2
mkdir $FEATURES/$FEATURESFNAME3
echo "extracting valid"
$TOOLS/extract_features models/a128googlenet/finetune100f3ksd/a128_googlenet_quick_iter_69500.caffemodel models/a128googlenet/deploy160g/valid.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/npy,$FEATURES/$FEATURESFNAME2/npy,$FEATURES/$FEATURESFNAME3/npy 13 lmdb GPU 1

echo " done valid"


FEATURES=data/alisc/gf100f3ksd695/query
#mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME1
mkdir $FEATURES/$FEATURESFNAME2
mkdir $FEATURES/$FEATURESFNAME3

echo "extracting featrue query"


$TOOLS/extract_features models/a128googlenet/finetune100f3ksd/a128_googlenet_quick_iter_69500.caffemodel models/a128googlenet/deploy160g/query.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/npy,$FEATURES/$FEATURESFNAME2/npy,$FEATURES/$FEATURESFNAME3/npy 41 lmdb GPU 1

echo " done "$i""
