import json
import os

from .Dataset import Dataset


class Criteo(Dataset):
    block_size = 2000000
    train_num_of_parts = 44
    test_num_of_parts = 7
    train_size = 86883012
    test_size = 12733031
    train_pos_samples = 0
    test_pos_samples = 0
    train_neg_samples = 0
    test_neg_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    num_fields = 39
    max_length = num_fields
    num_features = 1178909
    feat_names = ['num_%d' % i for i in range(13)]
    feat_names.extend(['cat_%d' % i for i in range(26)])
    feat_sizes = [4389, 8000, 329, 7432, 2646, 428, 233, 6301, 295, 11, 173, 176642,
                  585, 147117, 19845, 14830, 6916, 18687, 4, 6646, 1272, 46, 141085, 64381,
                  63692, 11, 2156, 7806, 61, 5, 928, 15, 147387, 116331, 145634, 57186, 9307, 63, 34]
    feat_min = [0, 4389, 12389, 12718, 20150, 22796, 23224, 23457, 29758, 30053, 30064, 30237, 206879, 207464, 354581,
                374426, 389256, 396172, 414859, 414863, 421509, 422781, 422827, 563912, 628293, 691985, 691996, 694152,
                701958, 702019, 702024, 702952, 702967, 850354, 966685, 1112319, 1169505, 1178812, 1178875]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Criteo-8d')
    raw_data_dir = os.path.join(data_dir, 'raw')
    feature_data_dir = os.path.join(data_dir, 'feature')
    hdf_data_dir = os.path.join(data_dir, 'hdf')

    def __init__(self, initialized=True):
        """
        collect meta information, and produce hdf files if not exists
        :param initialized: write feature and hdf files if True
        """
        self.initialized = initialized
        if not self.initialized:
            import h5py

            print('Got raw Criteo 8-day logs, initializing data set...')
            if self.max_length is None or self.num_features is None:
                print('Getting the maximum length and # features...')
                h5file = h5py.File(os.path.join(self.raw_data_dir, 'criteo'))
                train_length = h5file['train'].shape[1]
                test_length = h5file['test'].shape[1]
                print('train set:', h5file['train'].shape, 'test set:', h5file['test'].shape)
                self.max_length = max(train_length, test_length)
                self.num_features = sum(json.loads(h5file.attrs['sizes'])[1:])
            print('max length = %d, # features = %d' % (self.max_length, self.num_features))

            self.train_num_of_parts = self.raw_to_feature(key='train',
                                                          input_feat_file='train_input.txt',
                                                          output_feat_file='train_output.txt')
            self.feature_to_hdf(num_of_parts=self.train_num_of_parts,
                                file_prefix='train',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)
            self.test_num_of_parts = self.raw_to_feature(key='test',
                                                         input_feat_file='test_input.txt',
                                                         output_feat_file='test_output.txt')
            self.feature_to_hdf(num_of_parts=self.test_num_of_parts,
                                file_prefix='test',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)
        print('Got hdf Criteo-8d data set, getting metadata...')
        self.train_size, self.train_pos_samples, self.train_neg_samples, self.train_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'train', self.train_num_of_parts)
        self.test_size, self.test_pos_samples, self.test_neg_samples, self.test_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'test', self.test_num_of_parts)
        print('Initialization finished!')

    def raw_to_feature(self, key, input_feat_file, output_feat_file):
        print('Transferring raw', key, 'data into feature', key, 'data...')

        input_feat_file = os.path.join(self.feature_data_dir, input_feat_file)
        output_feat_file = os.path.join(self.feature_data_dir, output_feat_file)
        cur_part = 0
        if self.block_size is not None:
            fin = open(input_feat_file + '.part_' + str(cur_part), 'w')
            fout = open(output_feat_file + '.part_' + str(cur_part), 'w')
        else:
            fin = open(input_feat_file, 'w')
            fout = open(output_feat_file, 'w')

        import h5py

        h5file = h5py.File('../../Ads/APEXDatasets/criteo')
        line_no = 0
        size = 10000
        while True:
            df = h5file[key][line_no: line_no + size]
            X_i, y_i = df[:, 1:], df[:, 0]
            X_i += self.feat_min

            for i in range(X_i.shape[0]):
                fout.write(str(y_i[i]))
                fout.write('\n')
                fin.write(str(X_i[i, 0]))
                for j in range(1, X_i.shape[1]):
                    fin.write(',' + str(X_i[i, j]))
                fin.write('\n')

            line_no += y_i.shape[0]
            if self.block_size is not None and line_no % self.block_size == 0:
                fin.close()
                fout.close()
                print('part', cur_part, 'finish')
                cur_part += 1
                fin = open(input_feat_file + '.part_' + str(cur_part), 'w')
                fout = open(output_feat_file + '.part_' + str(cur_part), 'w')

            if len(df) < size:
                break

        fin.close()
        fout.close()
        return cur_part + 1
