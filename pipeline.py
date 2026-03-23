'''
Download and import the training and test images from MNIST and Fashion MNIST. The imported
data are required to be kept in NumPy array format. Complete the following tasks on both of the datasets.
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy as idx
from tensorflow.keras.datasets import mnist, fashion_mnist

#train_image_file = './datasets/mnist_images.idx'
#train_label_file = './datasets/mnist_labels.idx'
#train_images = idx.convert_from_file(train_image_file)
#train_labels = idx.convert_from_file(train_label_file)

# MNIST - 70,000 images (60k are for training) of handwritten digits from 0-9.
# Each sample is a 28x28 grayscale image

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(X_train_f, Y_train_f), (X_test_f, Y_test_f) = fashion_mnist.load_data()
#flatten all data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
Y_train = Y_train.reshape(Y_train.shape[0], -1)
Y_test = Y_test.reshape(Y_test.shape[0], -1)

X_train_f = X_train_f.reshape(X_train_f.shape[0], -1)
X_test_f = X_test_f.reshape(X_test_f.shape[0], -1)
Y_train_f = Y_train_f.reshape(Y_train_f.shape[0], -1)
Y_test_f = Y_test_f.reshape(Y_test_f.shape[0], -1)

print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))

plt.imshow(X_train_f[0])
plt.show()



#%% 
