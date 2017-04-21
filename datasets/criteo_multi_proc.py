import os

import numpy as np

from multi_proc import multi_proc

key = 'train'
number_of_part = 44

raw_data_dir = '../data/Criteo-8d/raw'
feature_data_dir = '../data/Criteo-8d/feature'
hdf_data_dir = '../data/Criteo-8d/hdf'
feature_input_file_name = key + '_input.txt'
feature_output_file_name = key + '_output.txt'
split_size = 2000000
max_length = 39
feat_sizes = [4389, 8000, 329, 7432, 2646, 428, 233, 6301, 295, 11, 173, 176642,
              585, 147117, 19845, 14830, 6916, 18687, 4, 6646, 1272, 46, 141085, 64381,
              63692, 11, 2156, 7806, 61, 5, 928, 15, 147387, 116331, 145634, 57186, 9307, 63, 34]
feat_min = [sum(feat_sizes[:i]) for i in range(max_length)]

print 'Transferring raw', key, 'data into feature', key, 'data...'

feature_input_file_name = os.path.join(feature_data_dir, feature_input_file_name)
feature_output_file_name = os.path.join(feature_data_dir, feature_output_file_name)

import h5py


def pre_proc(npart, **kwargs):
    for i in range(npart):
        yield {
            'part_num': i,
        }


def part_job(kwargs):
    cur_part = kwargs['part_num']
    h5file = h5py.File('/home/kevin/Dataset/Ads/APEXDatasets/criteo')
    d = h5file[key][split_size * cur_part: split_size * (cur_part + 1)]
    y = np.reshape(d[:, 0], [-1])
    X = d[:, 1:] + feat_min
    print 'part', cur_part
    print 'got X', X.shape
    print 'got y', y.shape
    fin = open(feature_input_file_name + '.part_' + str(cur_part), 'w')
    fout = open(feature_output_file_name + '.part_' + str(cur_part), 'w')

    for i in range(X.shape[0]):
        fout.write(str(y[i]))
        fout.write('\n')
        fin.write(str(X[i, 0]))
        for j in range(1, X.shape[1]):
            fin.write(',' + str(X[i, j]))
        fin.write('\n')

    fin.close()
    fout.close()


multi_proc(number_of_part, pre_proc, part_job)
