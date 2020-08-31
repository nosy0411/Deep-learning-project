import tensorflow as tf
# TensorFlow 관련 warning 표시 무시
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TensorFlow 정보 출력 무시
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'

# Using TensorFlow backend. 메시지 무시
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

# 시작
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import PIL.ImageOps

# 딥러닝 모델 불러들이기
model = load_model('./mnist_model.h5')

# 케라스로 이미지 불러들이기
hand_writing_number = image.load_img("./number_img.png",
                                     color_mode="grayscale",
                                     target_size=(28, 28),
                                     interpolation="bilinear")

inverted_image = PIL.ImageOps.invert(hand_writing_number)

hand_writing = np.array(inverted_image)

output = model.predict(hand_writing.reshape((1, 28, 28, -1)))

print(f"\n그림의 손글씨 숫자는 {np.argmax(output)} 입니다.")
