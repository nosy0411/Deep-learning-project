import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from IPython.display import display
import PIL

# How to check if the code is running on GPU or CPU?

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()