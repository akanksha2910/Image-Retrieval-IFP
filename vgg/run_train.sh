TOOLS=./build/tools

$TOOLS/caffe train  -solver=models/vgg/quick_solver.prototxt -weights=models/vgg/vgg_finetune_all_v4_iter_100000.caffemodel -gpu all  2>&1 | tee szm_vgg_2016_256
