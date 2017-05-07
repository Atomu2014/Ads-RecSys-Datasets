from __future__ import print_function
import time

from iPinYou import iPinYou
from Criteo import Criteo


def test_dataset(dataset, train_gen=False, val_gen=False, test_gen=False,
                 batch_size=10000, pos_ratio=0.001, val_ratio=0.05,
                 random_sample=True, split_fields=False, on_disk=True):
    dataset = dataset()
    dataset.summary()
    if train_gen:
        print('testing train generator...')
        train_gen = dataset.batch_generator(gen_type='train', batch_size=batch_size, pos_ratio=pos_ratio,
                                            val_ratio=val_ratio, random_sample=random_sample, split_fields=split_fields,
                                            on_disk=on_disk)
        total_num = int(dataset.train_size * (1 - val_ratio))
        num_of_batches = total_num / batch_size
        tic = time.time()
        for idx in range(num_of_batches):
            X, y = train_gen.next()
            if idx % 100 == 0:
                print(idx, num_of_batches)
                if split_fields:
                    print(len(X), y.shape)
                else:
                    print(X.shape, y.shape)
        toc = time.time()
        print('total train data generate time: ', (toc - tic))
    if val_gen:
        print('testing val generator...')
        val_gen = dataset.batch_generator(gen_type='valid', batch_size=batch_size, pos_ratio=pos_ratio,
                                          val_ratio=val_ratio, random_sample=random_sample, split_fields=split_fields,
                                          on_disk=on_disk)
        total_num = int(dataset.train_size * val_ratio)
        num_of_batches = total_num / batch_size
        tic = time.time()
        for idx in range(num_of_batches):
            X, y = val_gen.next()
            if idx % 100 == 0:
                print(idx, num_of_batches)
                if split_fields:
                    print(len(X), y.shape)
                else:
                    print(X.shape, y.shape)
        toc = time.time()
        print('total val data generate time: ', (toc - tic))
    if test_gen:
        print('testing test generator...')
        test_gen = dataset.batch_generator(gen_type='test', batch_size=batch_size, pos_ratio=pos_ratio,
                                           random_sample=random_sample, split_fields=split_fields, on_disk=on_disk)
        total_num = dataset.test_size
        num_of_batches = total_num / batch_size
        tic = time.time()
        for idx in range(num_of_batches):
            X, y = test_gen.next()
            if idx % 100 == 0:
                print(idx, num_of_batches)
                if split_fields:
                    print(len(X), y.shape)
                else:
                    print(X.shape, y.shape)
        toc = time.time()
        print('total test data generate time: ', (toc - tic))


# test_dataset(Criteo, True, True, True, random_sample=True, split_fields=False, on_disk=False)
test_dataset(iPinYou, True, True, True, random_sample=True, split_fields=False, on_disk=False)
