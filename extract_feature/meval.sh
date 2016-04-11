TOOLS=./build/tools
FEATURESG=data/alisc/gfea240k
FEATURES=data/alisc/gfea240k/eval
FEATURESNAME=inception_5b/output,inception_4d/output,loss2/fc_128,loss2/conv,loss2/ave_pool,loss1/fc_128,loss1/conv,loss1/ave_pool
FEATURESFNAME1=inception_5b_output
FEATURESFNAME2=inception_4d_output
FEATURESFNAME3=loss2_fc_128
FEATURESFNAME4=loss2_conv
FEATURESFNAME5=loss2_ave_pool
FEATURESFNAME6=loss1_fc_128
FEATURESFNAME7=loss1_conv
FEATURESFNAME8=loss1_ave_pool
mkdir $FEATURESG
mkdir $FEATURES
mkdir $FEATURES/$FEATURESFNAME1
mkdir $FEATURES/$FEATURESFNAME2
mkdir $FEATURES/$FEATURESFNAME3
mkdir $FEATURES/$FEATURESFNAME4
mkdir $FEATURES/$FEATURESFNAME5
mkdir $FEATURES/$FEATURESFNAME6
mkdir $FEATURES/$FEATURESFNAME7
mkdir $FEATURES/$FEATURESFNAME8

for ((i=0;i<63;i++))
do
echo "extracting featruev"$i""
$TOOLS/extract_features models/a128googlenet/finetuneB/a128_googlenet_quick_iter_14000.caffemodel models/a128googlenet/lmdb_deploy/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i",$FEATURES/$FEATURESFNAME3/"$i",$FEATURES/$FEATURESFNAME4/"$i",$FEATURES/$FEATURESFNAME5/"$i",$FEATURES/$FEATURESFNAME6/"$i",$FEATURES/$FEATURESFNAME7/"$i",$FEATURES/$FEATURESFNAME8/"$i" 500 lmdb GPU 1

echo " done "$i""
done

$i=63
echo "extracting featrue "$i""

$TOOLS/extract_features models/a128googlenet/finetuneB/a128_googlenet_quick_iter_14000.caffemodel models/a128googlenet/lmdb_deploy/eval_"$i".prototxt $FEATURESNAME $FEATURES/$FEATURESFNAME1/"$i",$FEATURES/$FEATURESFNAME2/"$i",$FEATURES/$FEATURESFNAME3/"$i",$FEATURES/$FEATURESFNAME4/"$i",$FEATURES/$FEATURESFNAME5/"$i",$FEATURES/$FEATURESFNAME6/"$i",$FEATURES/$FEATURESFNAME7/"$i",$FEATURES/$FEATURESFNAME8/"$i" 1193 lmdb GPU 1

echo " done "$i""
