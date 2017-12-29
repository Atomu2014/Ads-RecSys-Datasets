from __future__ import print_function

import time


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
