# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt

def create_weight(shape, kernel, is_comp=False):
    max_epsilon = np.min(np.abs(kernel))
    if is_comp:
        weights = np.zeros((shape, shape), dtype=np.float32)
    else:
        weights = np.random.randn(shape, shape) * 0.1 * max_epsilon
        weights = weights.astype(np.float32)

    for i in range(0, shape):
        index = 0
        for j in range(i, shape):
            if index < kernel.size:
                weights[j, i] = kernel[index]
                index += 1
    return weights

l_filter = np.array([-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304], dtype=np.float32)
h_filter = np.array([-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106], dtype=np.float32)

weights = create_weight(96, l_filter)
print(weights)
plt.imshow(weights)
plt.show()
