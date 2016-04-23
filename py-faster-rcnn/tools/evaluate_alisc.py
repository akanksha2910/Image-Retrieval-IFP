"""Evaluate mAP for all query images

For ALISC data

To evaluate performance, this requires following files:
1. ground_truth file: valid_fn
2. feature and image list for query images
3. feature and image list for eval images

optional: could evaluate performance of multiple feature combination

Data:
- data_root: the root folder for ALISC data & feature
- + eval_tags/valid_image.txt: the ground truth
- + query_features:
-   + MODEL_NAME
-     + feature.npy
-     + img_list.txt
- + eval_features:
-   + MODEL_NAME:
-     + feature.npy
-     + img_list.txt

To feed multiple models and use combination of those features:
--model_name MODEL_1, MODEL_2, ...
And use
--weights 0.5, 0.5 ...
to assign weights for different feature

By default, the similarity is calculated using cosine.
L2 distance is also implemented.

Copyright @ Xianming Liu, University of Illinois, Urbana-Champaign
"""
import heapq
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import argparse
import multiprocessing.pool as mp


def read_alisc_groundtruth(valid_fn):
    with open(valid_fn, 'r') as f:
        valid_img_list = f.readlines()
    valid_id_list = []
    ground_truth = []
    for i in range(len(valid_img_list)):
        linestr = valid_img_list[i]
        line_split = linestr.rstrip(';\r\n').split(',')
        query_id = line_split[0]
        valid_id_list.append(query_id)
        ground_truth.append(line_split[1].split(';'))
    return valid_id_list, ground_truth


def search_k_largest(vec, k):
    """Search K largest elements"""
    return heapq.nlargest(k, enumerate(vec), key=lambda x: x[1])


def search_k_smallest(vec, k):
    return heapq.nsmallest(k, enumerate(vec), key=lambda x: x[1])


def eval_ap(search_list, gt_list):
    """Calculate the Average precision for top k results

    Both search_list and gt_list are all list of strings, as eval image ids
    """
    num_hits = 0.0
    score = 0.0
    gt_list = [int(x) for x in gt_list]
    search_list = [int(x) for x in search_list]
    k = len(search_list)
    """
    print("Ground Truth:")
    print gt_list
    print("Searched Result:")
    print search_list
    """
    for i, p in enumerate(search_list):
        if p in gt_list:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            # score += 1
    return score / min(len(gt_list), k)


def read_feature(feature_dir):
    """Read features from csv file and IDs from txt file"""
    if not os.path.exists(os.path.join(feature_dir, 'imgIDs.txt')):
        raise
    # read features from either csv or npy files
    if os.path.exists(os.path.join(feature_dir, 'feature.csv')):
        # read feature from csv file
        df = pd.read_csv(os.path.join(feature_dir, 'feature.csv'),
                         header=None, engine='python')
        feature = df.values
    elif os.path.exists(os.path.join(feature_dir, 'feature.npy')):
        feature = np.load(os.path.join(feature_dir, 'feature.npy'))

    with open(os.path.join(feature_dir, 'imgIDs.txt'), 'r') as fp:
        imgIDs = fp.readlines()
        imgIDs = [x.rstrip('\n') for x in imgIDs]
    return imgIDs, feature


def calculate_distance_mat(x, y, distance_type='euclidean'):
    return pairwise_distances(x, y, metric=distance_type, n_jobs=-1)


def query_one_sample(distances, k=20, imgIDs=None):
    """Perform query / search top K for one sample
    Use the prcompted distances
    imgIDs is the list of eval imageID, which is used for
    evaluation compared with groundtruth
    """
    top_k = search_k_smallest(distances, k)
    if imgIDs is not None:
        top_k = [imgIDs[x[0]] for x in top_k]
    else:
        top_k = [x[0] for x in top_k]
    return top_k


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate algorithm based on features")
    parser.add_argument('--model_name', required=True, nargs='+',
                        help='Model Name')
    parser.add_argument('--root', default="../data/ALISC",
                        help="Root folder of all data. default in ../data/")
    parser.add_argument('--weights', nargs='+',
                        help="Weights for each model in multiple-model case")
    c = vars(parser.parse_args())
    c = vars(parser.parse_args())
    model_names = c['model_name']
    # read groundtruth
    DATA_ROOT = c['root']
    print("1. Reading groudtruth...")
    valid_fn = os.path.join(DATA_ROOT, 'eval_tags/valid_image.txt')
    valid_ids, gt = read_alisc_groundtruth(valid_fn)

    dis_mats = []
    for model_name in model_names:
        print("**********************Model {}********************".format(
            model_name))
        print("2. Reading features for query images for model [{}]...".format(
            model_name))
        valid_features = np.load(os.path.join(
            DATA_ROOT, 'query_features', model_name, 'feature.npy'
        ))
        print("3. Reading features for eval images for model [{}]...".format(
            model_name))
        eval_ids, eval_features = read_feature(
            os.path.join(DATA_ROOT, 'eval_features', model_name))

        print("4. Calculating Distances...")
        distance_type = 'cosine'
        dis_mat_ = calculate_distance_mat(
            valid_features, eval_features, distance_type=distance_type)
        dis_mats.append(dis_mat_)

    if c['weights'] is not None:
        weights = [float(x) for x in c['weights']]
    else:
        # take average weights
        weights = np.ones(len(model_names)) / len(model_names)
    print "Wegiths", weights
    # merge distance mats
    dis_mat = np.zeros(dis_mats[0].shape)
    for i in range(len(weights)):
        dis_mat += weights[i] * dis_mats[i]

    return_lists = {}
    knn = []
    aps = []
    print("5. Evaluate each images...")
    valid_MAP = 0

    pool = mp.Pool(16)
    results = [
        pool.apply_async(search_k_smallest, args=(
            dis_mat[i, ...].ravel(), 20)
        )
        for i in range(len(valid_ids))]
    for r in results:
        knn.append([x[0] for x in r.get()])

    for i in range(len(knn)):
        valid_id = valid_ids[i]
        top_k = [eval_ids[x] for x in knn[i]]
        return_lists[valid_id] = top_k
        ap = eval_ap(top_k, gt[i])
        print("ImageID: {} / AP = {}".format(valid_id, ap))
        valid_MAP += ap
        aps.append(ap)
    valid_MAP /= len(valid_ids)

    # save results
    with open('./{}_MAP.txt'.format("_".join(model_names)), 'w') as fp:
        fp.write('\n'.join([str(x) for x in aps]))
        fp.write('\n')
        fp.write('MAP: ')
        fp.write(str(valid_MAP))
    print("MAP= ", valid_MAP)

    with open('./{}_list.txt'.format("_".join(model_names)), 'w') as fp:
        # pickle.dump(return_lists, fp)
        for key, value in return_lists.iteritems():
            fp.write('{},{}\n'.format(key, ';'.join(value)))


if __name__ == '__main__':
    main()
