TOOLS=./build/tools
FEATURESG=data/alisc/gfea3B1k
FEATURES=data/alisc/gfea3B1k/eval
FEATURESNAME=inception_5b/output,pool5/7x7_s1_128
FEATURESFNAME1=inception_5b_output  
FEATURESFNAME2=pool5_7x7_s1_128
mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME1
mkdir $FEATURES/$FEATURESFNAME2

for ((i=0;i<63;i++))
do
echo "extracting featruev"$i""
$TOOLS/extract_features models/a128googlenet/finetuneB3/a128_googlenet_quick_iter_1000.caffemodel models/a128googlenet/lmdb_deploy/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i" 500 lmdb GPU 1

echo " done "$i""
done

$i=63
echo "extracting featrue "$i""

$TOOLS/extract_features models/a128googlenet/finetuneB3/a128_googlenet_quick_iter_1000.caffemodel models/a128googlenet/lmdb_deploy/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i" 1193 lmdb GPU 1

echo " done "$i""
