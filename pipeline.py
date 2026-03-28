import numpy as np
import time 
from tensorflow.keras.datasets import mnist, fashion_mnist
from pipeline_trainer import SVCPipelineTrainer
import task4_utility as t4

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


SEED = 96
#ITERATIONS = [1000, 3000, 8000]
ITERATIONS = [8000]
PCA_DIMS = [50,100,200]

KERNALS = ['linear', 'rbf', 'poly']
Cs = [2,0.1,4,0.1,0.5,1]
GAMMAs = ["scale", 0.005, 0.01, 0.01, 0.001, 0.0005]
DEGREEs = [2,2,1,2,3,3]
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
                             "poly"    : './results/fashion/poly.csv'},
              "bootstrap" : {"fashion" : './results/bootstrap/fashion.csv',
                             "mnist"   : './results/bootstrap/mnist.csv'}
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
                                   num_iter=ITERATIONS,
                                   data = mnist_data,
                                   pca_comps=PCA_DIMS, 
                                   fpaths=FILE_PATHS,
                                   cs=Cs,
                                   gammas=GAMMAs,
                                   degrees=DEGREEs,
                                   bootstrap=False)
mnist_trainer.run_all_variations()
                                   
'''fashion'''

fashion_data = [X_train_f, Y_train_f, X_test_f, Y_test_f]
fashion_trainer = SVCPipelineTrainer(folder_name='fashion', 
                                     kerns=KERNALS,
                                     data = fashion_data,
                                     pca_comps=PCA_DIMS, 
                                     num_iter=ITERATIONS, 
                                     fpaths=FILE_PATHS,
                                     cs=Cs,
                                     gammas=GAMMAs,
                                     degrees=DEGREEs,
                                     bootstrap=False)

fashion_trainer.run_all_variations()
'''#4'''

'''mnist bootstrap'''
best_kerns= ['linear','rbf','poly']
best_dims = [100,100,200] 
best_cs = [0.01, 4.0,0.1]
best_gammas = ['scale', 0.001,0.01]
best_degrees = [0, 0,2]

pipes = []
for i in range(len(best_dims)):
    pipe = Pipeline([("scaler",StandardScaler()),
                        ("dim_red",PCA(n_components=best_dims[i])),
                        ("model", SVC(C=best_cs[i], gamma=best_gammas[i],
                                      degree=best_degrees[i], kernel=best_kerns[i],
                                      max_iter=3000))],
                        verbose=True)
    pipes.append(pipe)


mnist_data = [X_train, Y_train, X_test, Y_test]
svc_bagged_data = t4.split_8(mnist_data, SEED)

x_train_subs = svc_bagged_data["x_trains"]
y_train_subs = svc_bagged_data["y_trains"]

start_time = time.time()

num_sets = 8
all_pipes = []

# train 8 copies of each of the 3 best models
for j in range(len(pipes)):
    for i in range(num_sets):
        X_sub = x_train_subs[i]
        y_sub = y_train_subs[i]

        pipe = clone(pipes[j])
        pipe.fit(X_sub, y_sub)
        all_pipes.append(pipe)


train_preds = []
test_preds = []

for pipe in all_pipes:
    train_preds.append(pipe.predict(X_train))
    test_preds.append(pipe.predict(X_test))

train_predictions = np.array(train_preds)
test_predictions = np.array(test_preds)

final_train_predictions = t4.majority_vote(train_predictions.T)
final_test_predictions = t4.majority_vote(test_predictions.T)

end_time = time.time()
total_time = end_time - start_time

train_score = np.mean(final_train_predictions == Y_train)
test_score = np.mean(final_test_predictions == Y_test)

for i in range(len(best_kerns)):
    print(f"dim_red=PCA(n_components={best_dims[i]}) | "
          f"num_iter=3000 | "
          f"kernal={best_kerns[i]} | "
          f"C={best_cs[i]} | "
          f"gamma={best_gammas[i]} | "
          f"degree={best_degrees[i]} | "
          f"time={total_time:.4f} | "
          f"train_score={train_score} | "
          f"test_score={test_score}")
    
    print("---------------------------------------- Complete! ----------------------------------------\n")


print("Bagged train score:", train_score)
print("Bagged test score:", test_score)
print("Bagged train error:", 1 - train_score)
print("Bagged test error:", 1 - test_score)

'''fashion bootstrapping'''

best_kerns= ['linear','rbf','poly']
best_dims = [100,200,200] 
best_cs = [0.005, 2.0,10]
best_gammas = ['scale', 0.0005,0.001]
best_degrees = [0, 0,2]

pipes = []
for i in range(len(best_dims)):
    pipe = Pipeline([("scaler",StandardScaler()),
                        ("dim_red",PCA(n_components=best_dims[i])),
                        ("model", SVC(C=best_cs[i], gamma=best_gammas[i],
                                      degree=best_degrees[i], kernel=best_kerns[i],
                                      max_iter=3000))],
                        verbose=True)
    pipes.append(pipe)


mnist_data = [X_train_f, Y_train_f, X_test_f, Y_test_f]
svc_bagged_data = t4.split_8(mnist_data, SEED)

x_train_subs = svc_bagged_data["x_trains"]
y_train_subs = svc_bagged_data["y_trains"]

start_time = time.time()

num_sets = 8
all_pipes = []

# train 8 copies of each of the 3 best models
for j in range(len(pipes)):
    for i in range(num_sets):
        X_sub = x_train_subs[i]
        y_sub = y_train_subs[i]

        pipe = clone(pipes[j])
        pipe.fit(X_sub, y_sub)
        all_pipes.append(pipe)


train_preds = []
test_preds = []

for pipe in all_pipes:
    train_preds.append(pipe.predict(X_train_f))
    test_preds.append(pipe.predict(X_test_f))

train_predictions = np.array(train_preds)
test_predictions = np.array(test_preds)

final_train_predictions = t4.majority_vote(train_predictions.T)
final_test_predictions = t4.majority_vote(test_predictions.T)

end_time = time.time()
total_time = end_time - start_time

train_score = np.mean(final_train_predictions == Y_train_f)
test_score = np.mean(final_test_predictions == Y_test_f)

for i in range(len(best_kerns)):
    print(f"dim_red=PCA(n_components={best_dims[i]}) | "
          f"num_iter=3000 | "
          f"kernal={best_kerns[i]} | "
          f"C={best_cs[i]} | "
          f"gamma={best_gammas[i]} | "
          f"degree={best_degrees[i]} | "
          f"time={total_time:.4f} | "
          f"train_score={train_score} | "
          f"test_score={test_score}")
    
    print("---------------------------------------- Complete! ----------------------------------------\n")


print("Bagged train score:", train_score)
print("Bagged test score:", test_score)
print("Bagged train error:", 1 - train_score)
print("Bagged test error:", 1 - test_score)
