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
# test_dataset(iPinYou, True, True, True, random_sample=True, split_fields=False, on_disk=False)

# data = iPinYou()
# gen = data.batch_generator(kwargs={'gen_type': 'train', 'batch_size': 10000, 'val_ratio': 0.01})
# i = 0
# for d in gen:
#     i += 1
#     if i % 100 == 0:
#         print(i)
#     if i == 200:
#         break
# # gen = gen.reset()
# i = 0
# for d in gen:
#     i += 1
#     if i % 100 == 0:
#         print(i)


class a:
    def __init__(self, b):
        self.b = b
    # def reset(self):
    #     return self.b.run()
    def __iter__(self):
        for i in self.b.__iter__():
            yield i
        # yield self.b.__iter__()

class b:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        for i in range(self.n):
            yield i
    def run(self):
        return a(self)

b_gen = b(10).run()
b_iter = b_gen.__iter__()
for i in b_iter:
    print(i)
# b_gen = b_gen.reset()
b_iter = b_gen.__iter__()
for i in b_iter:
    print(i)
