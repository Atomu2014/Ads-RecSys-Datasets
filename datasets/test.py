from __future__ import print_function

import time

from Criteo_Challenge import Criteo_Challenge

dataset = Criteo_Challenge(False)
exit(0)


from iPinYou import iPinYou

dataset = iPinYou(True)
print('total size', dataset.test_size)
param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': 1000,
    'num_workers': 2,
    'task_index': 0,
}
test_gen = dataset.batch_generator(param)

cnt = 0
for x, y in test_gen:
    if not cnt:
        print(x)
    cnt += x.shape[0]
print('worker0 read', cnt)

param['task_index'] = 1
test_gen = dataset.batch_generator(param)

cnt = 0
for x, y in test_gen:
    if not cnt:
        print(x)
    cnt += x.shape[0]
print('worker1 read', cnt)

exit(0)


# class A:
#     def __init__(self, b, kwargs):
#         self.b = b
#         self.kwargs = kwargs
#
#     def __iter__(self):
#         print('here')
#         for x in self.b.__iter__(self.kwargs):
#             yield x
#
#     def reset(self):
#         return self
#
#
# class B:
#     def __iter__(self, n):
#         for i in range(n):
#             yield i
#
#     def gen(self, n):
#         return A(self, n)
#
# b = B().gen(5)
#
# print('!!!')
#
# bb = iter(b.reset())
# for i in range(3):
#     # for x in b:
#     #     print(x)
#     for j in range(5):
#         print(bb.next())
#     bb = iter(b.reset())
#     print('<--', i, '-->')
#
#
# exit(0)

from Criteo_all import Criteo_all

d = Criteo_all(True, 9)

test_data_param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': 20000,
}

test_gen = d.batch_generator(test_data_param)

for i in range(3):
    flag = True
    for x in test_gen:
        if flag:
            print(x)
            flag = False
    print(i)

test_iter = iter(test_gen)
for i in range(3):
    try:
        flag = True
        while True:
            x = test_iter.next()
            if flag:
                print(x)
                flag = False
    except StopIteration, e:
        test_iter = iter(test_gen)
    print(i)
