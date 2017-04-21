import os

from Dataset import Dataset


class iPinYou(Dataset):
    block_size = 2000000
    train_num_of_parts = 8
    test_num_of_parts = 3
    train_size = 0
    test_size = 0
    train_pos_samples = 0
    test_pos_samples = 0
    train_neg_samples = 0
    test_neg_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    max_length = 16
    num_features = 937670
    feat_names = ['weekday', 'hour', 'IP', 'region', 'city', 'adexchange', 'domain',
                  'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
                  'creative', 'advertiser', 'useragent', 'slotprice']
    feat_min = [0, 8, 33, 704998, 705034, 705405, 705411, 756732, 937427, 937449, 937464, 937476, 937481, 937613,
                937623, 937664]
    feat_max = [7, 32, 704997, 705033, 705404, 705410, 756731, 937426, 937448, 937463, 937475, 937480, 937612, 937622,
                937663, 937669]
    feat_sizes = [feat_max[i] - feat_min[i] + 1 for i in range(max_length)]
    data_dir = os.path.dirname(os.path.dirname(__file__))
    raw_data_dir = data_dir + '../iPinYou-all/raw'
    feature_data_dir = data_dir + '../iPinYou-all/feature'
    hdf_data_dir = data_dir + '../iPinYou-all/hdf'

    def __init__(self, initialized=True, dir_path='../iPinYou-all', max_length=None, num_features=None,
                 block_size=2000000):
        """
        collect meta information, and produce hdf files if not exists
        :param initialized: write feature and hdf files if True
        :param dir_path: 
        :param max_length: 
        :param num_features: 
        :param block_size: 
        """
        self.initialized = initialized
        if not self.initialized:
            print('Got raw iPinYou data, initializing...')
            self.raw_data_dir = os.path.join(dir_path, 'raw')
            self.feature_data_dir = os.path.join(dir_path, 'feature')
            self.hdf_data_dir = os.path.join(dir_path, 'hdf')
            self.max_length = max_length
            self.num_features = num_features
            self.block_size = block_size
            if self.max_length is None or self.num_features is None:
                print('Getting the maximum length and # features...')
                min_train_length, max_train_length, max_train_feature = self.get_length_and_feature_number(
                    os.path.join(self.raw_data_dir, 'train.txt'))
                min_test_length, max_test_length, max_test_feature = self.get_length_and_feature_number(
                    os.path.join(self.raw_data_dir, 'test.txt'))
                self.max_length = max(max_train_length, max_test_length)
                self.num_features = max(max_train_feature, max_test_feature) + 1
            print('max length = %d, # features = %d' % (self.max_length, self.num_features))

            self.train_num_of_parts = self.raw_to_feature(raw_file='train.txt',
                                                          input_feat_file='train_input.txt',
                                                          output_feat_file='train_output.txt')
            self.feature_to_hdf(num_of_parts=self.train_num_of_parts,
                                file_prefix='train',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir,
                                input_columns=self.feat_names,
                                output_columns=['click'])
            self.test_num_of_parts = self.raw_to_feature(raw_file='test.txt',
                                                         input_feat_file='test_input.txt',
                                                         output_feat_file='test_output.txt')
            self.feature_to_hdf(num_of_parts=self.test_num_of_parts,
                                file_prefix='test',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir,
                                input_columns=self.feat_names,
                                output_columns=['click'])

        print('Got hdf iPinYou data set, getting metadata...')
        self.train_size, self.train_pos_samples, self.train_neg_samples, self.train_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'train', self.train_num_of_parts)
        self.test_size, self.test_pos_samples, self.test_neg_samples, self.test_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'test', self.test_num_of_parts)
        print('Initialization finished!')

    def raw_to_feature(self, raw_file, input_feat_file, output_feat_file):
        """
        Transfer the raw data to feature data. using static method is for consistence 
            with multi-processing version, which can not be packed into a class
        :param raw_file: The name of the raw data file.
        :param input_feat_file: The name of the feature input data file.
        :param output_feat_file: The name of the feature output data file.
        :return:
        """
        print('Transferring raw', raw_file, 'data into feature', raw_file, 'data...')
        raw_file = os.path.join(self.raw_data_dir, raw_file)
        feature_input_file_name = os.path.join(self.feature_data_dir, input_feat_file)
        feature_output_file_name = os.path.join(self.feature_data_dir, output_feat_file)
        line_no = 0
        cur_part = 0
        if self.block_size is not None:
            fin = open(feature_input_file_name + '.part_' + str(cur_part), 'w')
            fout = open(feature_output_file_name + '.part_' + str(cur_part), 'w')
        else:
            fin = open(feature_input_file_name, 'w')
            fout = open(feature_output_file_name, 'w')
        with open(raw_file, 'r') as rin:
            for line in rin:
                line_no += 1
                if self.block_size is not None and line_no % self.block_size == 0:
                    fin.close()
                    fout.close()
                    cur_part += 1
                    fin = open(feature_input_file_name + '.part_' + str(cur_part), 'w')
                    fout = open(feature_output_file_name + '.part_' + str(cur_part), 'w')

                fields = line.strip().split()
                y_i = fields[0]
                X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
                fout.write(y_i + '\n')
                first = True

                if len(X_i) > self.max_length:
                    X_i = X_i[:self.max_length]
                elif len(X_i) < self.max_length:
                    X_i.extend([self.num_features + 1] * (self.max_length - len(X_i)))

                for item in X_i:
                    if first:
                        fin.write(str(item))
                        first = False
                    else:
                        fin.write(',' + str(item))
                fin.write('\n')
        fin.close()
        fout.close()
        return cur_part + 1

    @staticmethod
    def get_length_and_feature_number(file_name):
        """
        Get the min_length max_length and max_feature of data.
        :param file_name: The file name of input data.
        :return: the tuple (min_length, max_length, max_feature)
        """
        max_length = 0
        min_length = 99999
        max_feature = 0
        line_no = 0
        with open(file_name) as fin:
            for line in fin:
                line_no += 1
                if line_no % 100000 == 0:
                    print('%d lines finished.' % (line_no))
                fields = line.strip().split()
                X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
                max_feature = max(max_feature, max(X_i))
                max_length = max(max_length, len(X_i))
                min_length = min(min_length, len(X_i))
        return min_length, max_length, max_feature
