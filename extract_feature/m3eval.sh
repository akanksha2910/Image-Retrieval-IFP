TOOLS=./build/tools
FEATURESG=data/alisc/gf100f3ksd695
FEATURES=data/alisc/gf100f3ksd695/eval
FEATURESNAME=loss1/max_pool,loss2/max_pool,pool5/max_7x7_s1_128
FEATURESFNAME1=loss1_max_pool 
FEATURESFNAME2=loss2_max_pool
FEATURESFNAME3=pool5_max_7x7_s1_128
mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME1
mkdir $FEATURES/$FEATURESFNAME2
mkdir $FEATURES/$FEATURESFNAME3
for ((i=0;i<63;i++))
do
echo "extracting featruev"$i""
$TOOLS/extract_features models/a128googlenet/finetune100f3ksd/a128_googlenet_quick_iter_69500.caffemodel models/a128googlenet/deploy160g/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i",$FEATURES/$FEATURESFNAME3/"$i" 500 lmdb GPU 1

echo " done "$i""
done

$i=63
echo "extracting featrue "$i""

$TOOLS/extract_features models/a128googlenet/finetune100f3ksd/a128_googlenet_quick_iter_69500.caffemodel models/a128googlenet/deploy160g/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i",$FEATURES/$FEATURESFNAME3/"$i" 1193 lmdb GPU 1

echo " done "$i""
