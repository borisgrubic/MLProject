import math

class kernel_wrapper(object):
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        self.kwargs = kwargs

    def __call__(self, x, y):
        return self.kernel(x, y, **self.kwargs)

    def __repr__(self):
        return 'kernel: %s, kwargs: %s' % (self.kernel.__name__, self.kwargs)

def linear(x, y):
    n = len(x)
    res = 0.0
    for i in range(n):
        res += x[i] * y[i]
    return res

def gaussian(x, y, gamma=1.0):
    n = len(x)
    res = 0.0
    for i in range(n):
        res += (x[i] - y[i]) ** 2
    res *= -gamma
    return math.exp(res)
