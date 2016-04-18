"""Training script of ALISC Classification on Faster-RCNN
"""

import sys
sys.path.append('../../tools')
import _init_paths
import caffe
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import argparse
import os.path as osp


def main():
    current_dir = osp.abspath(__file__)
    parser = argparse.ArgumentParser("Traing ALISC Clasification Network")
    parser.add_argument('--network_type', help='VGG16 or ZF', required=True)

    c = vars(parser.parse_args())

    network_type = c['network_type']

    print("Loading Configuration File...")
    cfg = cfg_from_file(osp.join(
        current_dir,
        '../../experiments/cfgs/alisc_classification.yml'))

    solver_fn = osp.join(
        current_dir,
        '../../models/alisc/{}/solver.pt'.format(network_type))

    solver = caffe.SGDSolver(solver_fn)

    print('Start Solving...')
    solver.solve()
