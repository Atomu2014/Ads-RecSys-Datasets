from __future__ import division
from __future__ import print_function

import sys
if sys.version.startswith('2'):
    import cPickle as pkl
else:
    import pickle as pkl
import os

import numpy as np
import pandas as pd

from .Dataset import Dataset, DatasetHelper


class Criteo_all(Dataset):
    block_size = 2000000
    num_of_days = 9
    log_files = None
    initialized = True
    num_fields = 39
    max_length = num_fields
    feat_names = ['num_%d' % i for i in range(13)]
    feat_names.extend(['cat_%d' % i for i in range(26)])
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Criteo-all')
    raw_data_dir = os.path.join(data_dir, 'raw')
    feature_data_dir = os.path.join(data_dir, 'feature')
    hdf_data_dir = os.path.join(data_dir, 'hdf')

    def __init__(self, initialized=True, num_of_days=9):
        self.initialized = initialized
        if num_of_days == 9:
            self.num_of_days = num_of_days
            self.log_files = ['day_%d' % i for i in range(13, 22)]
            self.prefix = '7day'
            self.num_of_parts = [7, 7, 7, 6, 6, 6, 6, 7, 7]
            self.test_hdf_files = []
            self.file_sizes = [12533985, 12603268, 12214878, 11941595, 10894422, 10034652, 10975618, 12618941, 12583580]
            self.feat_sizes = [3548, 7980, 292, 7323, 2574, 420, 229, 6139, 202, 12, 174, 165439, 520, 130136, 19649,
                               14680, 6829, 18362, 4, 6734, 1273, 47, 125880, 60458, 61714, 11, 2152, 7755, 61, 5, 912,
                               15, 130386, 105920, 128911, 53885, 9394, 64, 34]
            self.feat_min = [0, 3548, 11528, 11820, 19143, 21717, 22137, 22366, 28505, 28707, 28719, 28893, 194332,
                             194852, 324988, 344637, 359317, 366146, 384508, 384512, 391246, 392519, 392566, 518446,
                             578904, 640618, 640629, 642781, 650536, 650597, 650602, 651514, 651529, 781915, 887835,
                             1016746, 1070631, 1080025, 1080089]
            self.num_features = 1080123
        elif num_of_days == 16:
            self.num_of_days = num_of_days
            self.log_files = ['day_%d' % i for i in range(6, 22)]
            self.prefix = '14day'
            self.num_of_parts = [7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 6, 6, 6, 6, 7, 7]
            self.file_sizes = [12965061, 13313392, 12778263, 13147871, 12254823, 10804759, 11622235, 12533985,
                               12603268, 12214878, 11941595, 10894422, 10034652, 10975618, 12618941, 12583580]
            self.feat_sizes = [4005, 8001, 315, 7393, 2618, 424, 232, 6237, 250, 12, 175, 171612, 556, 130923, 20174,
                               15337, 6873, 18509, 4, 6829, 1281, 47, 129172, 62254, 63242, 11, 2155, 7901, 61, 5, 919,
                               15, 130518, 109302, 131187, 55490, 9564, 63, 34]
            self.feat_min = [0, 4005, 12006, 12321, 19714, 22332, 22756, 22988, 29225, 29475, 29487, 29662, 201274,
                             201830, 332753, 352927, 368264, 375137, 393646, 393650, 400479, 401760, 401807, 530979,
                             593233, 656475, 656486, 658641, 666542, 666603, 666608, 667527, 667542, 798060, 907362,
                             1038549, 1094039, 1103603, 1103666]
            self.num_features = 1103700
        else:
            print('invalid setting! num_of_days should be 9 or 16')
            exit(0)
        self.train_size = sum(self.file_sizes[:-2])
        self.valid_size = self.file_sizes[-2]
        self.test_size = self.file_sizes[-1]

        self.train_hdf_files = []
        for i in range(self.num_of_days - 2):
            for j in range(self.num_of_parts[i]):
                self.train_hdf_files.append(os.path.join(self.hdf_data_dir, '%s_%s_<>_part_%d.h5' %
                                                         (self.prefix, self.log_files[i], j)))
        self.valid_hdf_files = [
            os.path.join(self.hdf_data_dir, '%s_%s_<>_part_%d.h5' % (self.prefix, self.log_files[-2], j))
            for j in range(self.num_of_parts[-2])]
        self.test_hdf_files = [
            os.path.join(self.hdf_data_dir, '%s_%s_<>_part_%d.h5' % (self.prefix, self.log_files[-1], j))
            for j in range(self.num_of_parts[-1])]

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        if not self.initialized:
            # down sample
            for _f in self.log_files:
                self.down_sample(_f)

            # create feature map
            if num_of_days == 9:
                feat_map = pkl.load(open('../Criteo-all/raw/7day_feat_map.pkl', 'rb'))
            elif num_of_days == 16:
                feat_map = pkl.load(open('../Criteo-all/raw/14day_feat_map.pkl', 'rb'))
            else:
                print('invalid setting! num_of_days should be 9 or 16')
                exit(0)

            feat_sizes = []
            num_feat = []
            for i in range(13):
                kv = []
                for k, v in feat_map[i].iteritems():
                    if k == '':
                        kv.append([-1, v])
                    else:
                        kv.append([int(k), v])
                kv = sorted(kv, key=lambda x: x[0])
                kv = np.array(kv)
                _s = 0
                thresholds = []
                for j in range(len(kv) - 1):
                    _k, _v = kv[j]
                    _s += _v
                    if _s > 40:
                        thresholds.append(_k)
                        _s = 0
                thresholds = np.array(thresholds)
                num_feat.append(thresholds)
                feat_sizes.append(len(num_feat[i]) + 1)

            cat_feat = []
            for i in range(13, 39):
                cat_feat.append({})
                for k, v in feat_map[i].iteritems():
                    if v > 40:
                        cat_feat[i - 13][k] = len(cat_feat[i - 13])
                cat_feat[i - 13]['other'] = len(cat_feat[i - 13])
                feat_sizes.append(len(cat_feat[i - 13]))

            pkl.dump(num_feat, open(os.path.join(self.raw_data_dir, self.prefix + '_num_feat.pkl'), 'wb'))
            print('dump num_feat')
            pkl.dump(cat_feat, open(os.path.join(self.raw_data_dir, self.prefix + '_cat_feat.pkl'), 'wb'))
            print('dump cat_feat')

            # collect statistics
            feat_sizes = []
            num_feat = pkl.load(open(os.path.join(self.raw_data_dir, self.prefix + '_num_feat.pkl'), 'rb'))
            cat_feat = pkl.load(open(os.path.join(self.raw_data_dir, self.prefix + '_cat_feat.pkl'), 'rb'))
            for i in range(13):
                feat_sizes.append(len(num_feat[i]) + 1)
            for i in range(26):
                feat_sizes.append(len(cat_feat[i]))
            feat_min = [sum(feat_sizes[:i]) for i in range(39)]
            print(feat_sizes)
            print(feat_min)
            print(sum(feat_sizes))

            file_sizes = []
            for i, _f in enumerate(self.log_files):
                cnt = 0
                for j in range(self.num_of_parts[i]):
                    f_in = os.path.join(self.feature_data_dir, '%s_%s_output.part_%d' % (self.prefix, _f, j))
                    with open(f_in) as fin:
                        cnt += len(fin.readlines())
                file_sizes.append(cnt)
            print(file_sizes)

            # split into blocks and convert to index
            for _f in self.log_files[15:18]:
                cur_index = 0
                f_in = os.path.join(self.raw_data_dir, _f + '.sample')

                def output(cur_index, X, y):
                    if len(y) == 0:
                        return

                    f_out_x = os.path.join(self.feature_data_dir, '%s_%s_input.part_%d' % (self.prefix, _f, cur_index))
                    f_out_y = os.path.join(self.feature_data_dir, '%s_%s_output.part_%d' % (self.prefix, _f, cur_index))
                    for i in range(len(X)):
                        for j in range(13):
                            _v = int(X[i][j]) if X[i][j] != '' else -1
                            X[i][j] = len(np.where(num_feat[j] < _v)[0])
                        for j in range(13, 39):
                            _v = X[i][j]
                            if _v in cat_feat[j - 13]:
                                X[i][j] = cat_feat[j - 13][_v]
                            else:
                                X[i][j] = cat_feat[j - 13]['other']
                    for i in range(len(X)):
                        X[i] = ' '.join([str(X[i][j] + feat_min[j]) for j in range(39)])
                        X[i] = X[i] + '\n'
                    with open(f_out_y, 'a') as fout_y:
                        for _y in y:
                            fout_y.write(_y)
                    with open(f_out_x, 'a') as fout_x:
                        for _x in X:
                            fout_x.write(_x)

                with open(f_in) as fin:
                    cnt = 0
                    y = []
                    X = []
                    for line in fin:
                        line = line.strip().split('\t')
                        y.append(line[0] + '\n')
                        X.append(line[1:])
                        cnt += 1
                        if cnt % 100000 == 0:
                            print('processing', cnt, 'output', _f, 'part', cur_index)
                            output(cur_index, X, y)
                            y = []
                            X = []
                        if cnt % self.block_size == 0:
                            cur_index += 1
                    output(cur_index, X, y)

            # convert to hdf
            for i, _f in enumerate(self.log_files):
                self.feature_to_hdf(self.num_of_parts[i], self.prefix + '_' + _f, self.feature_data_dir,
                                    self.hdf_data_dir)

    def down_sample(self, f):
        f_in = os.path.join(self.raw_data_dir, f)
        f_out = f_in + '.sample'
        with open(f_in, 'r') as fin:
            neg_cnt = 0
            pos_cnt = 0
            for line in fin:
                if line[0] == '1':
                    pos_cnt += 1
                else:
                    neg_cnt += 1
        neg_threshold = pos_cnt * 1. / neg_cnt
        print('file', f, 'pos_cnt', pos_cnt, 'neg_cnt', neg_cnt, 'neg_threshold:', neg_threshold)
        with open(f_in, 'r') as fin:
            buf = []
            fout = open(f_out, 'w')
            fout.close()
            cnt = 0
            for line in fin:
                if line[0] == '1':
                    buf.append(line)
                elif np.random.random() < neg_threshold:
                    buf.append(line)
                cnt += 1
                if cnt % 1000000 == 0:
                    print(cnt)
                if len(buf) >= 100000:
                    with open(f_out, 'a') as fout:
                        for _l in buf:
                            fout.write(_l)
                    buf = []
            with open(f_out, 'a') as fout:
                for _l in buf:
                    fout.write(_l)
                buf = []

    def summary(self):
        print(self.__class__.__name__, 'data set summary:')
        print('num of days:', self.num_of_days)
        print('train set:', self.log_files[:-2], 'valid set:', self.log_files[-2], 'test set:', self.log_files[-1])
        print('train size:', sum(self.file_sizes[:-2]), 'valid size:', self.file_sizes[:-2], 'test size:',
              self.file_sizes[-1])
        print('input max length = %d, number of categories = %d' % (self.max_length, self.num_features))
        print('features\tmin_index\tsize')
        for i in range(self.max_length):
            print('%s\t%d\t%d' % (self.feat_names[i], self.feat_min[i], self.feat_sizes[i]))

    def _files_iter_(self, gen_type='train', shuffle_block=False):
        gen_type = gen_type.lower()
        if gen_type == 'train':
            hdf_files = self.train_hdf_files
        elif gen_type == 'valid':
            hdf_files = self.valid_hdf_files
        elif gen_type == 'test':
            hdf_files = self.test_hdf_files
        if shuffle_block:
            np.random.shuffle(hdf_files)
        for x in hdf_files:
            yield x.replace('<>', 'input'), x.replace('<>', 'output')

    def load_data(self, gen_type='train', num_workers=1, task_index=0):
        gen_type = gen_type.lower()

        if gen_type == 'train':
            if self.X_train and self.y_train:
                return
            num_of_parts = len(self.train_hdf_files)
        elif gen_type == 'valid':
            if self.X_valid and self.y_valid:
                return
            num_of_parts = len(self.valid_hdf_files)
        elif gen_type == 'test':
            if self.X_test and self.y_test:
                return
            num_of_parts = len(self.test_hdf_files)

        X_all = []
        y_all = []
        for hdf_in, hdf_out in self._files_iter_(gen_type, False):
            print(hdf_in.split('/')[-1], '/', num_of_parts, 'loaded')
            num_lines = pd.HDFStore(hdf_out, mode='r').get_storer('fixed').shape[0]
            one_piece = int(np.ceil(num_lines / num_workers))
            start = one_piece * task_index
            stop = one_piece * (task_index + 1)
            X_block = pd.read_hdf(hdf_in, mode='r', start=start, stop=stop).as_matrix()
            y_block = pd.read_hdf(hdf_out, mode='r', start=start, stop=stop).as_matrix()
            X_all.append(X_block)
            y_all.append(y_block)
        X_all = np.vstack(X_all)
        y_all = np.vstack(y_all)

        if gen_type == 'train':
            self.X_train = X_all
            self.y_train = y_all
            print('all train data loaded')
        elif gen_type == 'valid':
            self.X_valid = X_all
            self.y_valid = y_all
            print('all valid data loaded')
        elif gen_type == 'test':
            self.X_test = X_all
            self.y_test = y_all
            print('all test data loaded')

    def batch_generator(self, kwargs):
        return DatasetHelper(self, kwargs)

    def __iter__(self, gen_type='train', batch_size=None, shuffle_block=False, random_sample=False, split_fields=False,
                 on_disk=True, squeeze_output=True, num_workers=1, task_index=0, **kwargs):
        gen_type = gen_type.lower()

        def _iter_():
            if on_disk:
                print('on disk...')
                for hdf_X, hdf_y in self._files_iter_(gen_type=gen_type, shuffle_block=shuffle_block):
                    num_lines = pd.HDFStore(hdf_y, mode='r').get_storer('fixed').shape[0]
                    one_piece = int(np.ceil(num_lines / num_workers))
                    start = one_piece * task_index
                    stop = one_piece * (task_index + 1)
                    X_all = pd.read_hdf(hdf_X, mode='r', start=start, stop=stop).as_matrix()
                    y_all = pd.read_hdf(hdf_y, mode='r', start=start, stop=stop).as_matrix()
                    yield X_all, y_all, hdf_X
            else:
                print('in memory...')
                self.load_data(gen_type=gen_type, num_workers=num_workers, task_index=task_index)
                if gen_type == 'train':
                    yield self.X_train, self.y_train, gen_type
                elif gen_type == 'valid':
                    yield self.X_valid, self.y_valid, gen_type
                elif gen_type == 'test':
                    yield self.X_test, self.y_test, gen_type

        for X_all, y_all, block in _iter_():
            gen = self.generator(X_all, y_all, batch_size, shuffle=random_sample)
            for X, y in gen:
                if split_fields:
                    X = np.split(X, self.max_length, axis=1)
                    for i in range(self.max_length):
                        X[i] -= self.feat_min[i]
                if squeeze_output:
                    y = y.squeeze()
                yield X, y
