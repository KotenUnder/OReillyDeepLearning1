import numpy as np
import os
import matplotlib.pylab as plt
import NeuralNetwork as nn
import load_mnist_numbers
from PIL import Image
import pickle

PWD = os.path.dirname(__file__)
DATA_FOLDER_NAME = os.path.join(PWD, "data", "mnist_number")
MNIST_PICKLE_PATH = os.path.join(DATA_FOLDER_NAME, "mnist.pickle")

"""
Step Functionの動作
x = np.arange(-5.0, 5.0, 0.1)
y = nn.step_function_array(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
"""

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def load_mnist_pickle():
    with open(MNIST_PICKLE_PATH, "rb") as f:
        obj = pickle.load(f)
        return obj

def save_as_pickle(pickle_object_):
    with open(MNIST_PICKLE_PATH, "wb") as f:
        pickle.dump(pickle_object_, f)

"""
PILの動作確認
# PILで画像確認
result = load_mnist_numbers.load_mnist()
img = result[0][0][0]
label = result[0][1][0]

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)

print("done")
"""

# (train_x, train_l), (verify_x, verify_l) =

