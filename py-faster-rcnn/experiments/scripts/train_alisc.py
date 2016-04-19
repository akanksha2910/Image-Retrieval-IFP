"""Training script of ALISC Classification on Faster-RCNN
"""

import os.path as osp
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
import sys
sys.path.append(osp.join(ROOT_DIR, 'tools'))
import _init_paths
import caffe
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import argparse


def main():
    parser = argparse.ArgumentParser("Traing ALISC Clasification Network")
    parser.add_argument('--network_type', help='VGG16 or ZF', required=True)
    parser.add_argument('--model', help="Pretrained Model, could be empty")
    c = vars(parser.parse_args())

    network_type = c['network_type']

    print("Loading Configuration File...")
    cfg = cfg_from_file(osp.join(
        ROOT_DIR,
        'experiments', 'cfgs', 'alisc_classification.yml'))

    solver_fn = osp.join(
        ROOT_DIR,
        'models', 'alisc', network_type, 'solver.pt')

    solver = caffe.SGDSolver(solver_fn)

    if c['model']:
        weights_fn = c['model']
    else:
        weights_fn = osp.join(
            ROOT_DIR, 'data', 'faster_rcnn_models',
            '{}_faster_rcnn_final.caffemodel'.format(network_type))
    if not osp.exists(weights_fn):
        print("Pretrained Model {} does not exists".format(weights_fn))
        raise

    print("Loading pretrained model from {}".format(weights_fn))
    solver.net.copy_from(weights_fn)

    print('Start Solving...')
    solver.solve()

if __name__ == '__main__':
    main()
