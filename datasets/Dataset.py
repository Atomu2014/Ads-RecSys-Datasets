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
    num_features = 0
    feat_names = None
    feat_min = None
    feat_sizes = None
    raw_data_dir = None
    feature_data_dir = None
    hdf_data_dir = None

    X_train = None
    y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None

    def raw_to_feature(self, **kwargs):
        """
        this method should be override
        :return: 
        """
        pass

    @staticmethod
    def feature_to_hdf(num_of_parts, file_prefix, feature_data_dir, hdf_data_dir, input_columns,
                       output_columns):
        """
        convert lib-svm feature files into hdf5 files (tables). using static method is for consistence 
            with multi-processing version, which can not be packed into a class
        :param num_of_parts: 
        :param file_prefix: a prefix is suggested to identify train/test/valid/..., e.g. file_prefix='train'
        :param feature_data_dir: 
        :param hdf_data_dir: 
        :param input_columns: to name the columns of inputs, e.g. input_columns=['city', 'IP', ...]
        :param output_columns: to name the columns of output(s), e.g. output_columns=['click']
        :return: 
        """
        print('Transferring feature', file_prefix, 'data into hdf data and save hdf', file_prefix, 'data...')
        for idx in range(num_of_parts):
            _X = pd.read_csv(os.path.join(feature_data_dir, file_prefix + '_input.txt.part_' + str(idx)),
                             names=input_columns, dtype=np.int32)
            _y = pd.read_csv(os.path.join(feature_data_dir, file_prefix + '_output.txt.part_' + str(idx)),
                             names=output_columns, dtype=np.int32)
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
            _y = pd.read_hdf(os.path.join(hdf_data_dir, file_prefix + '_output_part_' + str(part) + '.h5'))
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
        print('train set: ', self.train_size)
        print('\tpositive samples: ', self.train_pos_samples)
        print('\tnegative samples: ', self.train_neg_samples)
        print('\tpositive ratio: ', self.train_pos_ratio)
        print('test size:', self.test_size)
        print('\tpositive samples: ', self.test_pos_samples)
        print('\tnegative samples: ', self.test_neg_samples)
        print('\tpositive ratio: ', self.test_pos_ratio)
        print('input max length = %d, number of categories = %d' % (self.max_length, self.num_features))
        print('features\tmin_index\tsize')
        for i in range(len(self.feat_names)):
            print('%s\t%d\t%d' % (self.feat_names[i], self.feat_min[i], self.feat_sizes[i]))

    def _files_iter_(self, gen_type='train', num_of_parts=None, shuffle_block=False):
        """
        iterate among hdf files(blocks). when the whole data set is finished, the iterator restarts 
            from the beginning, thus the data stream will never stop
        :param gen_type: could be 'train', 'valid', or 'test'. when gen_type='train' or 'valid', 
            this file iterator will go through the train set
        :param num_of_parts: 
        :param shuffle_block: shuffle block files at every round
        :return: input_hdf_file_name, output_hdf_file_name, finish_flag
        """
        gen_type = gen_type.lower()
        if gen_type == 'train' or gen_type == 'valid':
            file_prefix = 'train'
        elif gen_type == 'test':
            file_prefix = 'test'
        if num_of_parts == 0:
            yield os.path.join(self.hdf_data_dir, file_prefix + '_input.h5'), \
                  os.path.join(self.hdf_data_dir, file_prefix + '_output.h5'),
        else:
            if num_of_parts is None:
                if gen_type == 'train' or gen_type == 'valid':
                    num_of_parts = self.train_num_of_parts
                elif gen_type == 'test':
                    num_of_parts = self.test_num_of_parts
            parts = np.arange(num_of_parts)
            if shuffle_block:
                for i in range(int(shuffle_block)):
                    np.random.shuffle(parts)
            for i, p in enumerate(parts):
                yield os.path.join(self.hdf_data_dir, file_prefix + '_input_part_' + str(p) + '.h5'), \
                      os.path.join(self.hdf_data_dir, file_prefix + '_output_part_' + str(p) + '.h5'),

    def load_data(self, gen_type='train', num_of_parts=None, random_sample=False, val_ratio=None):
        if val_ratio is None:
            val_ratio = 0.0
        if num_of_parts is None:
            if gen_type == 'train' or gen_type == 'valid':
                if self.train_num_of_parts is not None:
                    num_of_parts = self.train_num_of_parts
            elif gen_type == 'test':
                if self.test_num_of_parts is not None:
                    num_of_parts = self.test_num_of_parts
        X_all = []
        y_all = []
        for hdf_in, hdf_out in self._files_iter_(gen_type, num_of_parts, False):
            print(hdf_in.split('/')[-1], '/', num_of_parts, 'loaded')
            X_block = pd.read_hdf(hdf_in).as_matrix()
            y_block = pd.read_hdf(hdf_out).as_matrix()
            X_all.append(X_block)
            y_all.append(y_block)
        X_all = np.vstack(X_all)
        y_all = np.vstack(y_all)
        if random_sample:
            idx = np.arange(X_all.shape[0])
            for i in range(int(random_sample)):
                np.random.shuffle(idx)
            X_all = X_all[idx]
            y_all = y_all[idx]
        if gen_type == 'train' or gen_type == 'valid':
            sep = int(X_all.shape[0] * val_ratio)
            self.X_valid = X_all[:sep]
            self.y_valid = y_all[:sep]
            self.X_train = X_all[sep:]
            self.y_train = y_all[sep:]
            print('all train/valid data loaded')
        elif gen_type == 'test':
            self.X_test = X_all
            self.y_test = y_all
            print('all test data loaded')

    def batch_generator(self, kwargs):
        return DatasetHelper(self, kwargs)

    def __iter__(self, gen_type='train', batch_size=None, pos_ratio=None, num_of_parts=None, val_ratio=None,
                 random_sample=False, shuffle_block=False, split_fields=False, on_disk=True, split_pos_neg=True,
                 squeeze_output=False):
        """
        :param gen_type: 'train', 'valid', or 'test'.  the valid set is partitioned from train set dynamically
        :param batch_size: 
        :param pos_ratio: default value is decided by the dataset, which means you don't want to change is
        :param num_of_parts: 
        :param val_ratio: fraction of valid set from train set
        :param random_sample: if True, will shuffle
        :param shuffle_block: shuffle file blocks at every round
        :param split_fields: if True, returned values will be independently indexed, else using unified index
        :param on_disk: if true iterate on disk, random_sample in block, if false iterate in mem, random_sample on all data
        :param split_pos_neg: if true every block will be split into pos and neg, every batch is a mixture of the two
        :return: 
        """
        if batch_size is None:
            batch_size = max(int(1 / self.train_pos_ratio), int(1 / self.test_pos_ratio)) + 1
        if val_ratio is None:
            val_ratio = 0.0
        gen_type = gen_type.lower()
        if num_of_parts is None:
            if gen_type == 'train' or gen_type == 'valid':
                if self.train_num_of_parts is not None:
                    num_of_parts = self.train_num_of_parts
            elif gen_type == 'test':
                if self.test_num_of_parts is not None:
                    num_of_parts = self.test_num_of_parts
        if on_disk:
            print('on disk...')
            for hdf_in, hdf_out in self._files_iter_(gen_type, num_of_parts, shuffle_block):
                number_of_lines = pd.HDFStore(hdf_in).get_storer('fixed').shape[0]

                if gen_type == 'train':
                    start = int(number_of_lines * val_ratio)
                    stop = None
                elif gen_type == 'valid':
                    start = None
                    stop = int(number_of_lines * val_ratio)
                else:
                    start = stop = None
                X_all = pd.read_hdf(hdf_in, start=start, stop=stop).as_matrix()
                y_all = pd.read_hdf(hdf_out, start=start, stop=stop).as_matrix()

                if split_pos_neg:
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
                            pos_X, pos_y, pos_finished = pos_gen.next()
                            neg_X, neg_y, neg_finished = neg_gen.next()
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
                            print('finish block', hdf_in)
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
        else:
            print('in mem...')
            self.load_data(gen_type, num_of_parts, random_sample, val_ratio)
            if gen_type == 'train':
                X_all = self.X_train
                y_all = self.y_train
            elif gen_type == 'valid':
                X_all = self.X_valid
                y_all = self.y_valid
            elif gen_type == 'test':
                X_all = self.X_test
                y_all = self.y_test

            if split_pos_neg:
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
                        pos_X, pos_y, pos_finished = pos_gen.next()
                        neg_X, neg_y, neg_finished = neg_gen.next()
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
        num_of_batches = int(np.ceil(X.shape[0] / batch_size))
        finished = False
        sample_index = np.arange(X.shape[0])
        if shuffle:
            for i in range(int(shuffle)):
                np.random.shuffle(sample_index)
        assert X.shape[0] > 0
        for i in range(num_of_batches):
            batch_index = sample_index[batch_size * i: batch_size * (i + 1)]
            if batch_index.size < batch_size:
                remain = batch_size - batch_index.size
                batch_index = np.append(batch_index, sample_index[:remain])
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            yield X_batch, y_batch, finished

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
