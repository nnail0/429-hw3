import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pipeline_trainer

SEED = 96
#ITERATIONS = [1000, 3000, 8000]
ITERATIONS = [8000]
PCA_DIMS = [50,100,200]

KERNALS = ['linear', 'rbf', 'poly']
Cs = [2,0.1,4,0.1,0.5,1]
GAMMAs = ["scale", 0.005, 0.01, 0.01, 0.001, 0.0005]
DEGREEs = [2,2]
ALL_HPs = [Cs, GAMMAs, DEGREEs]

FILE_PATHS = {"mnist" :     {"pca50"   : './results/mnist/pca50.csv',
                             "pca100"  : './results/mnist/pca100.csv',
                             "pca200"  : './results/mnist/pca200.csv',
                             "lda"     : './results/mnist/lda.csv',
                             "linear"  : './results/mnist/linear.csv',
                             "rbf"     : './results/mnist/rbf.csv',
                             "poly"    : './results/mnist/poly.csv'},
              "fashion" :   {"pca50"   : './results/fashion/pca50.csv',
                             "pca100"  : './results/fashion/pca100.csv',
                             "pca200"  : './results/fashion/pca200.csv',
                             "lda"     : './results/fashion/lda.csv',
                             "linear"  : './results/fashion/linear.csv',
                             "rbf"     : './results/fashion/rbf.csv',
                             "poly"    : './results/fashion/poly.csv'}
              "bootstrap" : {"fashion" : './results/bootstrap/fashion.csv',
                             "mnist"   : './results/bootstrap/fashion.csv'}
              }

'''#1'''
# MNIST - 70,000 images (60k are for training) of handwritten digits from 0-9.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(X_train_f, Y_train_f), (X_test_f, Y_test_f) = fashion_mnist.load_data()

'''#2'''
#flatten all data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train_f = X_train_f.reshape(X_train_f.shape[0], -1)
X_test_f = X_test_f.reshape(X_test_f.shape[0], -1)

'''#3'''
'''mnist'''

mnist_data = [X_train, Y_train, X_test, Y_test]
mnist_trainer = SVCPipelineTrainer(kerns=KERNALS,
                                   folder_name='mnist', 
                                   pca_comps=PCA_DIMS, 
                                   num_iter=ITERATIONS, 
                                   fpaths=FILE_PATHS,
                                   cs=best_cs,
                                   gammas=best_gammas,
                                   degrees=best_degrees)
mnist_trainer.svc_train()
                                   
'''fashion'''

fashion_data = [X_train_f, Y_train_f, X_test_f, Y_test_f]
fashion_trainer = SVCPipelineTrainer(folder_name='fashion', 
                                     kerns=KERNALS, 
                                     pca_comps=PCA_DIMS, 
                                     num_iter=ITERATIONS, 
                                     fpaths=FILE_PATHS,
                                     cs=Cs,
                                     gammas=GAMMAs,
                                     degrees=DEGREEs)

'''#4'''
all_kernals = [50,100,200]
best_cs = [0.1, 2.0, 0.1]
best_gammas = [0, 0.005, 0.01]
best_degrees = [0, 0, 2]

mnist_data = [X_train, Y_train, X_test, Y_test]
mnist_trainer = SVCPipelineTrainer(kerns=KERNALS,
                                   folder_name='mnist', 
                                   pca_comps=PCA_DIMS, 
                                   num_iter=ITERATIONS, 
                                   fpaths=FILE_PATHS,
                                   cs=best_cs,
                                   gammas=best_gammas,
                                   degrees=best_degrees)

mnist_trainer.run_all_variations()

