import os

import numpy as np
import pandas as pd


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
            part_pos_num = _y.loc[_y.ix[:, 0] == 1].shape[0]
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

    def _iterate_hdf_files_(self, gen_type='train', num_of_parts=None, shuffle_block=False):
        """
        iterate among hdf files(blocks). when the whole data set is finished, the iterator restarts 
            from the beginning, thus the data stream will never stop
        :param gen_type: could be 'train', 'valid', or 'test'. when gen_type='train' or 'valid', 
            this file iterator will go through the train set
        :param num_of_parts: 
        :param shuffle_block: shuffle block files at every round
        :return: input_hdf_file_name, output_hdf_file_name, finish_flag
        """
        if gen_type.lower() == 'train' or gen_type.lower() == 'valid':
            file_prefix = 'train'
        elif gen_type.lower() == 'test':
            file_prefix = 'test'
        if num_of_parts is None:
            yield os.path.join(self.hdf_data_dir, file_prefix + '_input.h5'), \
                  os.path.join(self.hdf_data_dir, file_prefix + '_output.h5'), True
        else:
            parts = np.arange(num_of_parts)
            while True:
                if shuffle_block:
                    for i in range(int(shuffle_block)):
                        np.random.shuffle(parts)
                for p in parts:
                    yield os.path.join(self.hdf_data_dir, file_prefix + '_input_part_' + str(p) + '.h5'), \
                          os.path.join(self.hdf_data_dir, file_prefix + '_output_part_' + str(p) + '.h5'), False

    def batch_generator(self, gen_type='train', batch_size=None, pos_ratio=None, num_of_parts=None, val_ratio=None,
                        random_sample=False, shuffle_block=False, split_fields=False):
        """
        :param gen_type: 'train', 'valid', or 'test'.  the valid set is partitioned from train set dynamically
        :param batch_size: 
        :param pos_ratio: default value is decided by the dataset, which means you don't want to change is
        :param num_of_parts: 
        :param val_ratio: fraction of valid set from train set
        :param random_sample: if True, will shuffle
        :param shuffle_block: shuffle file blocks at every round
        :param split_fields: if True, returned values will be independently indexed, else using unified index
        :return: 
        """
        if pos_ratio is None:
            pos_ratio = self.train_pos_ratio
        if batch_size is None:
            batch_size = max(int(1 / self.train_pos_ratio), int(1 / self.test_pos_ratio)) + 1
        if val_ratio is None:
            val_ratio = 0.0
        if num_of_parts is None:
            if gen_type.lower() == 'train' or gen_type.lower() == 'valid':
                if self.train_num_of_parts is not None:
                    num_of_parts = self.train_num_of_parts
            elif gen_type.lower() == 'test':
                if self.test_num_of_parts is not None:
                    num_of_parts = self.test_num_of_parts

        for hdf_in, hdf_out, ignore_finish in self._iterate_hdf_files_(gen_type, num_of_parts, shuffle_block):
            number_of_lines = pd.HDFStore(hdf_in).get_storer('fixed').shape[0]

            if gen_type.lower() == 'train':
                start = int(number_of_lines * val_ratio)
                stop = None
            elif gen_type.lower() == 'val':
                start = None
                stop = int(number_of_lines * val_ratio)
            else:
                start = stop = None
            X_all = pd.read_hdf(hdf_in, start=start, stop=stop).as_matrix()
            y_all = pd.read_hdf(hdf_out, start=start, stop=stop).as_matrix()

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
            pos_finished = False
            neg_finished = False
            while ignore_finish or (not pos_finished and not neg_finished):
                pos_X, pos_y, pos_finished = pos_gen.next()
                neg_X, neg_y, neg_finished = neg_gen.next()
                X = np.append(pos_X, neg_X, axis=0)
                y = np.append(pos_y, neg_y, axis=0)
                if split_fields:
                    X = np.split(X, self.max_length, axis=1)
                    for i in range(self.max_length):
                        X[i] -= self.feat_min[i]
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
        number_of_batches = np.ceil(X.shape[0] / batch_size)
        counter = 0
        finished = False
        sample_index = np.arange(X.shape[0])
        if shuffle:
            for i in range(int(shuffle)):
                np.random.shuffle(sample_index)
        assert X.shape[0] > 0
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            while batch_index.size != batch_size:
                remain = batch_size - batch_index.size
                batch_index = np.append(batch_index, sample_index[:remain])
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch, finished
            if counter == number_of_batches:
                counter = 0
                finished = True

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
