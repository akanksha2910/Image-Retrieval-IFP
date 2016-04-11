TOOLS=./build/tools
FEATURESG=data/alisc/gfea240k
FEATURES=data/alisc/gfea240k/eval
FEATURESNAME=inception_5b/output
FEATURESFNAME=inception_5b_output
mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME

for ((i=0;i<63;i++))
do
echo "extracting featruev"$i""
$TOOLS/extract_features models/a128googlenet/finetune/a128_googlenet_quick_iter_240000.caffemodel models/a128googlenet/lmdb_deploy/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/"$i" 500 lmdb GPU 1

echo " done "$i""
done


echo "extracting featrue 63"

$TOOLS/extract_features models/a128googlenet/finetune/a128_googlenet_quick_iter_240000.caffemodel models/a128googlenet/lmdb_deploy/eval_63.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/63 1193 lmdb GPU 1

echo " done 63"
