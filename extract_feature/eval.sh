TOOLS=./build/tools
FEATURESG=data/alisc/gfea100R37
FEATURES=data/alisc/gfea100R37/eval
FEATURESNAME=pool5/max_7x7_s1_128
FEATURESFNAME=pool5_max_7x7_s1_128
mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME

for ((i=0;i<63;i++))
do
echo "extracting featruev"$i""
$TOOLS/extract_features models/a128googlenet/finetune100R37k/a128_googlenet_quick_iter_12000.caffemodel models/a128googlenet/deploy160/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/"$i" 500 lmdb GPU 1

echo " done "$i""
done


echo "extracting featrue 63"

$TOOLS/extract_features models/a128googlenet/finetune100R37k/a128_googlenet_quick_iter_12000.caffemodel models/a128googlenet/deploy160/eval_63.prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME/63 1193 lmdb GPU 1

echo " done 63"
