import os

from Dataset import Dataset


class Avazu(Dataset):
    block_size = 2000000
    train_num_of_parts = 17
    test_num_of_parts = 5
    train_size = 32343173
    test_size = 8085794
    train_pos_samples = 0
    test_pos_samples = 0
    train_neg_samples = 0
    test_neg_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    num_fields = 24
    max_length = num_fields
    num_features = 645195
    feat_names = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                  'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16',
                  'C17', 'C18', 'C19', 'C20', 'C21', 'mday', 'hour', 'wday']
    feat_sizes = [7, 7, 3135, 3487, 24, 4002, 252, 28, 101449, 523672, 5925, 5, 4, 2417, 8, 9, 426, 4, 67, 166, 60, 10,
                  24, 7]
    feat_min = [sum(feat_sizes[:i]) for i in range(max_length)]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Avazu')
    raw_data_dir = os.path.join(data_dir, 'raw')
    feature_data_dir = os.path.join(data_dir, 'feature')
    hdf_data_dir = os.path.join(data_dir, 'hdf')

    def __init__(self, initialized=True):
        self.initialized = initialized
        if not self.initialized:
            print('Got raw Avazu data, initializing...')
            print('max length = %d, # feature = %d' % (self.max_length, self.num_features))
            self.train_num_of_parts = self.raw_to_feature(raw_file='avazu.tr.svm',
                                                          input_feat_file='train_input.txt',
                                                          output_feat_file='train_output.txt')
            self.feature_to_hdf(num_of_parts=self.train_num_of_parts,
                                file_prefix='train',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)
            self.test_num_of_parts = self.raw_to_feature(raw_file='avazu.te.svm',
                                                         input_feat_file='test_input.txt',
                                                         output_feat_file='test_output.txt')
            self.feature_to_hdf(num_of_parts=self.test_num_of_parts,
                                file_prefix='test',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)

        print('Got hdf Avazu data set, getting metadata...')
        self.train_size, self.train_pos_samples, self.train_neg_samples, self.train_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'train', self.train_num_of_parts)
        self.test_size, self.test_pos_samples, self.test_neg_samples, self.test_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'test', self.test_num_of_parts)
        print('Initialization finished!')

    def raw_to_feature(self, raw_file, input_feat_file, output_feat_file):
        print('Transferring raw', raw_file, 'data into feature', raw_file, 'data...')
        raw_file = os.path.join(self.raw_data_dir, raw_file)
        feat_fin_name = os.path.join(self.feature_data_dir, input_feat_file)
        feat_fout_name = os.path.join(self.feature_data_dir, output_feat_file)
        line_no = 0
        cur_part = 0
        if self.block_size is not None:
            fin = open(feat_fin_name + '.part_' + str(cur_part), 'w')
            fout = open(feat_fout_name + '.part_' + str(cur_part), 'w')
        else:
            fin = open(feat_fin_name, 'w')
            fout = open(feat_fout_name, 'w')
        with open(raw_file, 'r') as rin:
            for line in rin:
                line_no += 1
                if self.block_size is not None and line_no % self.block_size == 0:
                    fin.close()
                    fout.close()
                    cur_part += 1
                    fin = open(feat_fin_name + '.part_' + str(cur_part), 'w')
                    fout = open(feat_fout_name + '.part_' + str(cur_part), 'w')

                fields = line.strip().split()
                y_i = fields[0]
                X_i = map(lambda x: x.split(':')[0], fields[1:])
                fout.write(y_i + '\n')
                fin.write(','.join(X_i))
                fin.write('\n')
        fin.close()
        fout.close()
        return cur_part + 1
