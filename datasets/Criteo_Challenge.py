from __future__ import division
from __future__ import print_function

import cPickle as pkl
import os

from Dataset import Dataset


class Criteo_Challenge(Dataset):
    block_size = 2000000
    log_files = None
    initialized = True
    train_num_of_parts = 23
    test_num_of_parts = 4
    num_fields = 69
    max_length = num_fields
    num_features = 1010469
    feat_names = []
    feat_sizes = [63, 113, 126, 51, 224, 149, 100, 80, 104, 9, 32, 58, 82, 1457, 555, 176373, 129683, 305, 19, 11887,
                  632, 3, 41738, 5170, 175446, 3170, 27, 11356, 165602, 10, 4641, 2030, 4, 172761, 18, 15, 57903, 86,
                  44549, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                  128, 128, 128, 128, 128, 128, 128, 128, 126, 128, 128]
    feat_min = [0, 63, 176, 302, 353, 577, 726, 826, 906, 1010, 1019, 1051, 1109, 1191, 2648, 3203, 179576, 309259,
                309564, 309583, 321470, 322102, 322105, 363843, 369013, 544459, 547629, 547656, 559012, 724614, 724624,
                729265, 731295, 731299, 904060, 904078, 904093, 961996, 962082, 1006631, 1006759, 1006887, 1007015,
                1007143, 1007271, 1007399, 1007527, 1007655, 1007783, 1007911, 1008039, 1008167, 1008295, 1008423,
                1008551, 1008679, 1008807, 1008935, 1009063, 1009191, 1009319, 1009447, 1009575, 1009703, 1009831,
                1009959, 1010087, 1010213, 1010341]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Criteo-Challenge')
    raw_data_dir = os.path.join(data_dir, 'raw')
    feature_data_dir = os.path.join(data_dir, 'feature')
    hdf_data_dir = os.path.join(data_dir, 'hdf')

    def __init__(self, initialized=True):
        self.initialized = initialized
        self.train_hdf_files = [os.path.join(self.hdf_data_dir, 'train_<>_part_%d.h5' % j) for j in
                                range(self.train_num_of_parts)]
        self.test_hdf_files = [os.path.join(self.hdf_data_dir, 'test_<>_part_%d.h5' % j) for j in
                               range(self.test_num_of_parts)]

        if not self.initialized:
            feat_map = [{} for i in range(self.num_fields)]
            for f in ['train.ffm', 'test.ffm']:
                with open(os.path.join(self.raw_data_dir, f)) as fin:
                    for line in fin:
                        fields = line.strip().split()[1:]
                        for i in range(self.num_fields):
                            feat = fields[i].split(':')[1]
                            if feat not in feat_map[i]:
                                feat_map[i][feat] = len(feat_map[i])
            pkl.dump(feat_map, open(os.path.join(self.raw_data_dir, 'feat_map.pkl'), 'wb'))
            print('dump feat map')

            self.feat_sizes = [len(x) for x in feat_map]
            self.feat_min = [sum(self.feat_sizes[:i]) for i in range(self.num_fields)]
            self.num_features = sum(self.feat_sizes)
            print('feat_sizes', self.feat_sizes)
            print('feat_min', self.feat_min)
            print('num_features', self.num_features)

            # split into blocks and convert to index
            for f in ['train', 'test']:
                import numpy as np
                cur_index = 0
                f_in = os.path.join(self.raw_data_dir, f + '.ffm')

                def output(cur_index, X, y):
                    if len(y) == 0:
                        return

                    f_out_x = os.path.join(self.feature_data_dir, '%s_input.part_%d' % (f, cur_index))
                    f_out_y = os.path.join(self.feature_data_dir, '%s_output.part_%d' % (f, cur_index))
                    X = [' '.join([str(X[i][j] + self.feat_min[j]) for j in range(self.num_fields)]) for i in
                         range(len(X))]
                    if f == 'train':
                        ind = np.arange(len(X))
                        np.random.shuffle(ind)
                        X = np.array(X)[ind]
                        y = np.array(y)[ind]
                    with open(f_out_y, 'a') as fout_y:
                        for _y in y:
                            fout_y.write(_y + '\n')
                    with open(f_out_x, 'a') as fout_x:
                        for _x in X:
                            fout_x.write(_x + '\n')

                with open(f_in) as fin:
                    cnt = 0
                    y = []
                    X = []
                    for line in fin:
                        fields = line.strip().split()
                        y.append(fields[0] + '\n')
                        fields = [x.split(':')[1] for x in fields[1:]]
                        X.append([feat_map[i][fields[i]] for i in range(self.num_fields)])
                        cnt += 1
                        if cnt % self.block_size == 0:
                            print('processing', cnt, 'output', f, 'part', cur_index)
                            output(cur_index, X, y)
                            y = []
                            X = []
                            cur_index += 1
                    output(cur_index, X, y)

            # convert to hdf
            self.feature_to_hdf(self.train_num_of_parts, 'train', self.feature_data_dir, self.hdf_data_dir)
            self.feature_to_hdf(self.test_num_of_parts, 'test', self.feature_data_dir, self.hdf_data_dir)
