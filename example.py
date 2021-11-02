import numpy as np
import imageio

import decimator

# load a NumPy array
data = np.load('example.npy')

data_ = decimator.decimate(data, threshold=4.0)

print(data.shape, data.dtype)
print(data_.shape, data_.dtype)

# save as a NumPy array
np.save('example_result.npy', data_)

# save as a gif for visualization
imageio.mimsave('example_result.gif', data_)
