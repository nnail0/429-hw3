'''
Download and import the training and test images from MNIST and Fashion MNIST. The imported
data are required to be kept in NumPy array format. Complete the following tasks on both of the datasets.
'''

#%%
import numpy as np
import matplotlib as plt
import idx2numpy as idx
from tensorflow.keras.datasets import mnist

#train_image_file = './datasets/mnist_images.idx'
#train_label_file = './datasets/mnist_labels.idx'
#train_images = idx.convert_from_file(train_image_file)
#train_labels = idx.convert_from_file(train_label_file)

# MNIST - 70,000 images (60k are for training) of handwritten digits from 0-9.
# Each sample is a 28x28 grayscale image

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))



#%% 
