TOOLS=./build/tools
FEATURESG=data/alisc/gfeaB5L65k10k
FEATURES=data/alisc/gfeaB5L65k10k/eval
FEATURESNAME=loss1/max_pool,loss2/max_pool
FEATURESFNAME1=loss1_max_pool 
FEATURESFNAME2=loss2_max_pool
mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME1
mkdir $FEATURES/$FEATURESFNAME2

for ((i=0;i<63;i++))
do
echo "extracting featruev"$i""
$TOOLS/extract_features models/a128googlenet/finetuneB5L65k/a128_googlenet_quick_iter_10000.caffemodel models/a128googlenet/deploy/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i" 500 lmdb GPU 1

echo " done "$i""
done

$i=63
echo "extracting featrue "$i""

$TOOLS/extract_features models/a128googlenet/finetuneB5L65k/a128_googlenet_quick_iter_10000.caffemodel models/a128googlenet/deploy/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i" 1193 lmdb GPU 1

echo " done "$i""
