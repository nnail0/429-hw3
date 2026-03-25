# %% [markdown]
# # Pipeline for Midterm
# 
# 1. (2 points) Download and import the training and test images from MNIST and Fashion MNIST. The imported
# data are required to be kept in NumPy array format. Complete the following tasks on both of the datasets.

# %%
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy as idx
from tensorflow.keras.datasets import mnist, fashion_mnist

# %% [markdown]
# 2. (2 points) Perform a data format transformation by flattening each image to a 1-D NumPy array. You may
# use the NumPy function reshape.

# %%
# credit DigitalOcean
(train_X, train_y), (test_X, test_y) = mnist.load_data()
(train_X_f, train_y_f), (test_X_f, test_y_f) = fashion_mnist.load_data()

# flatten:


train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
train_y = train_y.reshape(train_y.shape[0])
test_X = test_X.reshape((test_X.shape[0],  test_X.shape[1] * test_X.shape[2]))
test_y = test_y.reshape(test_y.shape[0])

train_X_f = train_X_f.reshape((train_X_f.shape[0], train_X_f.shape[1] * train_X_f.shape[2]))
train_y_f = train_y_f.reshape(train_y_f.shape[0])
test_X_f = test_X_f.reshape((test_X_f.shape[0],  test_X_f.shape[1] * test_X_f.shape[2]))
test_y_f = test_y.reshape(test_y_f.shape[0])


# %% [markdown]
# #### 3. (11 points) Make a machine learning pipeline using scikit-learn to integrate the following steps:
# 
# 3.1 (1 point) Standardize the (flattened) samples. Notice that the preprocess mapping is computed based
# on the training data, but the same transformation should also be applied to the test data.

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

standard_pl : Pipeline
fashion_pl : Pipeline

sc = StandardScaler()


# %% [markdown]
# 3.2 (4 points) Dimensionality reduction on the data. You are required to use two ways: Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA), to reduce the number of features of the data. The original dimensionality is 784, you are required to consider the reduced dimensionalities: 50, 100 and 200 for PCA. Similar to the previous task, the reduction mappings should be derived from the training set but should also be used to compress the test data.

# %%
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pca_50 = PCA(n_components=50)
pca_100 = PCA(n_components=100)
pca_200 = PCA(n_components=200)

lda = LinearDiscriminantAnalysis()



# %% [markdown]
# 3.3 (6 points) Build a Support Vector Classifier (SVC) with a kernel for classifying the compressed data.
# 
# You should use the scikit-learn SVC class. You are required to consider the three kernels along with their hyperparameters:
# 
# - ’linear’ - Linear kernel, the only hyperparameter is C.
# - ’rbf’ - Radial basis function kernel, the hyperparameters are C and gamma.
# - ’poly’ - Polynomial kernel, the hyperparameters are C, gamma and degree.
# 
# You are required to tune the hyperparameters and choose the best setting for each kernel. When tuning the parameters, you need to measure the prediction error on both training set and test set.

# %%
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
import time


test_pipe = Pipeline([("scaler", sc), ("dim_red", pca_50), ("model", SVC(C = 0.01, kernel='linear', max_iter=1000))], verbose= True)
start_time = time.time()
test_pipe.fit(X = train_X, y = train_y)
time_taken = time.time() - start_time


print("score: ", test_pipe.score(test_X, test_y))
print("time: ", time_taken)

# In this block, we will need to test each combo. 
# Check the discord for helpful info from other people that have done similar stuff in the past. 


# params: C
# linear_svc = SVC(kernel='linear')

# params: C, gamma
# rbf_svc = SVC(kernel='rbf')

# params: C, gamma, degree
# poly_svc = SVC(kernel='poly')




# %% [markdown]
# ## Results from testing each:
# 
# PCA and LDA: compare time cost of training (incl. dimensionality reduction) and test error. 
# Three-kernel comparison: allowed to use PCA only. Training time (for SVC model only) and the prediction error. 
# 
# # Kernel Comparison - MNIST
# Linear, 50:
# 
# 
# Linear, 100:
# 
# 
# Linear, 200:
# 
# 
# RBF, 50:
# 
# 
# RBF, 100:
# 
# 
# RBF, 200:
# 
# 
# 
# Poly, 50:
# 
# 
# Poly, 100:
# 
# 
# Poly, 200: 
# 
# 
# # Kernel Comparison - Fashion MNIST
# Linear, 50:
# 
# 
# Linear, 100:
# 
# 
# Linear, 200:
# 
# 
# RBF, 50:
# 
# 
# RBF, 100:
# 
# 
# RBF, 200:
# 
# 
# 
# Poly, 50:
# 
# 
# Poly, 100:
# 
# 
# Poly, 200: 
# 
# 
# 


