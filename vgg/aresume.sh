TOOLS=./build/tools

$TOOLS/caffe train -snapshot=models/vgg/finetuneL140/szm_iter_5500.solverstate -solver=models/vgg/solver.prototxt -gpu all  2>&1 | tee szm_vgg_112L140R55

