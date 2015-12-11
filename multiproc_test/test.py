__author__ = 'Lenski Lab'
from multiprocessing import Process

def f(name):
    print 'hello', name
def func1(message):
    print message

def func2(message):
    print message

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    proc1 = Process(target=func1, args=('function one',))
    proc2 = Process(target=func2, args=('function two',))

    proc1.start()
    p.start()
    proc2.start()
    p.join()