from __future__ import print_function

import os

import numpy as np
import pandas as pd


class DatasetHelper:
    def __init__(self, dataset, kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

    def __iter__(self):
        for x in self.dataset.__iter__(**self.kwargs):
            yield x

    @property
    def batch_size(self):
        return self.kwargs['batch_size']

    @property
    def gen_type(self):
        return self.kwargs['gen_type']

    @property
    def val_ratio(self):
        return self.kwargs['val_ratio']


class Dataset:
    """
    block_size, train_num_of_parts, test_num_of_parts:
        raw data files will be partitioned into 'num_of_parts' blocks, each block has 'block_size' samples
    size, samples, ratio:
        train_size = train_pos_samples + train_neg_samples
        test_size = test_pos_sample + test_neg_samples
        train_pos_ratio = train_pos_samples / train_size
        test_neg_ratio = test_pos_samples / test_size
    initialized: decide whether to process (into hdf) or not
    features:
        max_length: different from # fields, in case some field has more than one value
        num_features: dimension of whole feature space
        feat_names: used as column names when creating hdf5 files
            num_of_fields = len(feat_names)
        feat_min: sometimes independent feature maps are needed, i.e. each field has an independent feature map 
            starting at index 0, feat_min is used to increment the index of feature maps and produce a unified 
            index feature map
        feat_sizes: sizes of feature maps
            feat_min[i] = sum(feat_sizes[:i])
    dirs:
        raw_data_dir: the original data is stored at raw_data_dir
        feature_data_dir: raw_to_feature() will process raw data and produce libsvm-format feature files,
            and feature engineering is done here
        hdf_data_dir: feature_to_hdf() will convert feature files into hdf5 tables, according to block_size
    """
    block_size = None
    train_num_of_parts = 0
    test_num_of_parts = 0
    train_size = 0
    test_size = 0
    train_pos_samples = 0
    test_pos_samples = 0
    train_neg_samples = 0
    test_neg_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    max_length = 0
    num_fields = 0
    num_features = 0
    feat_names = None
    feat_min = None
    feat_sizes = None
    raw_data_dir = None
    feature_data_dir = None
    hdf_data_dir = None

    X_train = None
    y_train = None
    X_test = None
    y_test = None

    def raw_to_feature(self, **kwargs):
        """
        this method should be override
        :return: 
        """
        pass

    @staticmethod
    def feature_to_hdf(num_of_parts, file_prefix, feature_data_dir, hdf_data_dir):
        """
        convert lib-svm feature files into hdf5 files (tables). using static method is for consistence 
            with multi-processing version, which can not be packed into a class
        :param num_of_parts: 
        :param file_prefix: a prefix is suggested to identify train/test/valid/..., e.g. file_prefix='train'
        :param feature_data_dir: 
        :param hdf_data_dir: 
        :return: 
        """
        print('Transferring feature', file_prefix, 'data into hdf data and save hdf', file_prefix, 'data...')
        for idx in range(num_of_parts):
            _X = pd.read_csv(os.path.join(feature_data_dir, file_prefix + '_input.part_' + str(idx)),
                             dtype=np.int32, delimiter=' ', header=None)
            _y = pd.read_csv(os.path.join(feature_data_dir, file_prefix + '_output.part_' + str(idx)),
                             dtype=np.int32, delimiter=' ', header=None)
            _X.to_hdf(os.path.join(hdf_data_dir, file_prefix + '_input_part_' + str(idx) + '.h5'), 'fixed')
            _y.to_hdf(os.path.join(hdf_data_dir, file_prefix + '_output_part_' + str(idx) + '.h5'), 'fixed')
            print('part:', idx, _X.shape, _y.shape)

    @staticmethod
    def bin_count(hdf_data_dir, file_prefix, num_of_parts):
        """
        count positive/negative samples
        :param hdf_data_dir: 
        :param file_prefix: see this param in feature_to_hdf()
        :param num_of_parts: 
        :return: size of a dataset, positive samples, negative samples, positive ratio
        """
        size = 0
        num_of_pos = 0
        num_of_neg = 0
        for part in range(num_of_parts):
            _y = pd.read_hdf(os.path.join(hdf_data_dir, file_prefix + '_output_part_' + str(part) + '.h5'), mode='r')
            part_pos_num = _y.loc[_y.iloc[:, 0] == 1].shape[0]
            part_neg_num = _y.shape[0] - part_pos_num
            size += _y.shape[0]
            num_of_pos += part_pos_num
            num_of_neg += part_neg_num
        pos_ratio = 1.0 * num_of_pos / (num_of_pos + num_of_neg)
        return size, num_of_pos, num_of_neg, pos_ratio

    def summary(self):
        """
        summarize the data set.
        :return: 
        """
        print(self.__class__.__name__, 'data set summary:')
        print('train set:', self.train_size)
        print('\tpositive samples:', self.train_pos_samples)
        print('\tnegative samples:', self.train_neg_samples)
        print('\tpositive ratio:', self.train_pos_ratio)
        print('test size:', self.test_size)
        print('\tpositive samples:', self.test_pos_samples)
        print('\tnegative samples:', self.test_neg_samples)
        print('\tpositive ratio:', self.test_pos_ratio)
        print('input max length = %d, number of categories = %d' % (self.max_length, self.num_features))
        print('features\tmin_index\tsize')
        for i in range(len(self.feat_names)):
            print('%s\t%d\t%d' % (self.feat_names[i], self.feat_min[i], self.feat_sizes[i]))

    def _files_iter_(self, gen_type='train', shuffle_block=False):
        """
        iterate among hdf files(blocks). when the whole data set is finished, the iterator restarts 
            from the beginning, thus the data stream will never stop
        :param gen_type: could be 'train', 'valid', or 'test'. when gen_type='train' or 'valid', 
            this file iterator will go through the train set
        :param shuffle_block: shuffle block files at every round
        :return: input_hdf_file_name, output_hdf_file_name, finish_flag
        """
        gen_type = gen_type.lower()
        if gen_type == 'train' or gen_type == 'valid':
            hdf_files = [os.path.join(self.hdf_data_dir, 'train_<>_part_%d.h5' % i)
                         for i in range(self.train_num_of_parts)]
        elif gen_type == 'test':
            hdf_files = [os.path.join(self.hdf_data_dir, 'test_<>_part_%d.h5' % i)
                         for i in range(self.test_num_of_parts)]
        if shuffle_block:
            np.random.shuffle(hdf_files)
        for f in hdf_files:
            yield f.replace('<>', 'input'), f.replace('<>', 'output')

    def load_data(self, gen_type='train', num_workers=1, task_index=0):
        gen_type = gen_type.lower()

        if gen_type == 'train' or gen_type == 'valid':
            if self.X_train and self.y_train:
                return
            num_of_parts = self.train_num_of_parts
        elif gen_type == 'test':
            if self.X_test and self.y_test:
                return
            num_of_parts = self.test_num_of_parts

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

        if gen_type == 'train' or gen_type == 'valid':
            self.X_train = X_all
            self.y_train = y_all
            print('all train/valid data loaded')
        elif gen_type == 'test':
            self.X_test = X_all
            self.y_test = y_all
            print('all test data loaded')

    def batch_generator(self, kwargs):
        return DatasetHelper(self, kwargs)

    def __iter__(self, gen_type='train', batch_size=None, pos_ratio=None, val_ratio=0.0, shuffle_block=False,
                 random_sample=False, split_fields=False, on_disk=True, squeeze_output=True, num_workers=1,
                 task_index=0):
        """
        :param gen_type: 'train', 'valid', or 'test'.  the valid set is partitioned from train set dynamically
        :param batch_size: 
        :param pos_ratio: default value is decided by the dataset, which means you don't want to change is
        :param val_ratio: fraction of valid set from train set
        :param random_sample: if True, will shuffle
        :param shuffle_block: shuffle file blocks at every round
        :param split_fields: if True, returned values will be independently indexed, else using unified index
        :param on_disk: if true iterate on disk, random_sample in block, if false iterate in mem, random_sample on all data
        :return: 
        """
        gen_type = gen_type.lower()

        def _iter_():
            if on_disk:
                print('on disk...')
                for hdf_in, hdf_out in self._files_iter_(gen_type=gen_type, shuffle_block=shuffle_block):
                    num_lines = pd.HDFStore(hdf_in, mode='r').get_storer('fixed').shape[0]
                    if gen_type == 'train':
                        start = int(num_lines * val_ratio)
                        stop = num_lines
                    elif gen_type == 'valid':
                        start = 0
                        stop = int(num_lines * val_ratio)
                    else:
                        start = 0
                        stop = num_lines
                    one_piece = int(np.ceil((stop - start)/ num_workers))
                    start = start + one_piece * task_index
                    stop = start + one_piece * (task_index + 1)
                    X_all = pd.read_hdf(hdf_in, mode='r', start=start, stop=stop).as_matrix()
                    y_all = pd.read_hdf(hdf_out, mode='r', start=start, stop=stop).as_matrix()
                    yield X_all, y_all, hdf_in
            else:
                print('in mem...')
                self.load_data(gen_type=gen_type, num_workers=num_workers, task_index=task_index)
                if gen_type == 'train' or gen_type == 'valid':
                    sep = int(len(self.X_train) * val_ratio)
                    if gen_type == 'train':
                        X_all = self.X_train[:sep]
                        y_all = self.y_train[:sep]
                    else:
                        X_all = self.X_train[sep:]
                        y_all = self.y_train[sep:]
                elif gen_type == 'test':
                    X_all = self.X_test
                    y_all = self.y_test
                yield X_all, y_all, 'all'

        for X_all, y_all, block in _iter_():
            if pos_ratio:
                X_pos, y_pos, X_neg, y_neg = self.split_pos_neg(X_all, y_all)
                number_of_pos = X_pos.shape[0]
                number_of_neg = X_neg.shape[0]
                if pos_ratio is None:
                    pos_ratio = 1.0 * number_of_pos / (number_of_pos + number_of_neg)
                if number_of_pos <= 0 or number_of_neg <= 0:
                    raise Exception('Invalid partition')
                pos_batchsize = int(batch_size * pos_ratio)
                neg_batchsize = batch_size - pos_batchsize
                if pos_batchsize <= 0 or neg_batchsize <= 0:
                    raise Exception('Invalid positive ratio.')
                pos_gen = self.generator(X_pos, y_pos, pos_batchsize, shuffle=random_sample)
                neg_gen = self.generator(X_neg, y_neg, neg_batchsize, shuffle=random_sample)
                while True:
                    try:
                        pos_X, pos_y = pos_gen.next()
                        neg_X, neg_y = neg_gen.next()
                        X = np.append(pos_X, neg_X, axis=0)
                        y = np.append(pos_y, neg_y, axis=0)
                        if split_fields:
                            X = np.split(X, self.max_length, axis=1)
                            for i in range(self.max_length):
                                X[i] -= self.feat_min[i]
                        if squeeze_output:
                            y = y.squeeze()
                        yield X, y
                    except StopIteration, e:
                        print('finish', block)
                        break
            else:
                gen = self.generator(X_all, y_all, batch_size, shuffle=random_sample)
                for X, y in gen:
                    if split_fields:
                        X = np.split(X, self.max_length, axis=1)
                        for i in range(self.max_length):
                            X[i] -= self.feat_min[i]
                    if squeeze_output:
                        y = y.squeeze()
                    yield X, y

    @staticmethod
    def generator(X, y, batch_size, shuffle=True):
        """
        should be accessed only in private
        :param X: 
        :param y: 
        :param batch_size: 
        :param shuffle: 
        :return: 
        """
        num_of_batches = int(np.ceil(X.shape[0] * 1.0 / batch_size))
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(sample_index)
        assert X.shape[0] > 0
        for i in range(num_of_batches):
            batch_index = sample_index[batch_size * i: batch_size * (i + 1)]
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            yield X_batch, y_batch

    @staticmethod
    def split_pos_neg(X, y):
        """
        should be access only in private
        :param X: 
        :param y: 
        :return: 
        """
        posidx = (y == 1).reshape(-1)
        X_pos, y_pos = X[posidx], y[posidx]
        X_neg, y_neg = X[~posidx], y[~posidx]
        return X_pos, y_pos, X_neg, y_neg

    def __str__(self):
        return self.__class__.__name__
