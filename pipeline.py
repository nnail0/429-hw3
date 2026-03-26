'''
Download and import the training and test images from MNIST and Fashion MNIST. The imported
data are required to be kept in NumPy array format. Complete the following tasks on both of the datasets.
'''

#%% Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist, fashion_mnist

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

PCA_DIMS = [50,100,200]
KERNALS = ['linear', 'rbf', 'poly']
ITERATIONS = [2000,5000,10000]
Cs = [10, 1, 0.1, 0.01, 0.001, 0.0001]
GAMMAs = ["scale", "auto", 0.1, 0.01, 0.001, 0.0001]
DEGREEs = [1,2,3,5,8,10]
FILE_PATHS = {"mnist" :   {"pca50" : './results/mnist/pca50.csv',
                          "pca100"  : './results/mnist/pca100.csv',
                          "pca200"  : './results/mnist/pca200.csv',
                          "lda"  : './results/mnist/lda.csv',
                          "linear"  : './results/mnist/linear.csv',
                          "rbf"  : './results/mnist/rbf.csv',
                          "poly"  : './results/mnist/poly.csv'},
              "fashion" : {"pca50" : './results/fashion/pca50.csv',
                          "pca100"  : './results/fashion/pca100.csv',
                          "pca200"  : './results/fashion/pca200.csv',
                          "lda"  : './results/fashion/lda.csv',
                          "linear"  : './results/fashion/linear.csv',
                          "rbf"  : './results/fashion/rbf.csv',
                          "poly"  : './results/fashion/poly.csv'}
              }

#%% #1
'''
Data Extraction
'''

# MNIST - 70,000 images (60k are for training) of handwritten digits from 0-9.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(X_train_f, Y_train_f), (X_test_f, Y_test_f) = fashion_mnist.load_data()

#%% #2
#flatten all data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train_f = X_train_f.reshape(X_train_f.shape[0], -1)
X_test_f = X_test_f.reshape(X_test_f.shape[0], -1)

# 3.1 Standardize the flatten samples
'''
Preprocessing Pipeline
'''
#TODO should we introduce Cross-validation?
standard_pl : Pipeline
fashion_pl : Pipeline
sc = StandardScaler()



# 3.3 SVC : Compare 3 Kernals
#%% 

def save_n_print_results(results, file_path):
    df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)
    df.to_csv(file_path, index=False)
    print(df.to_string(index=False))

def plot_t_vs_acc(times, scores, num_iter):
    '''
    Params : times (1D float array) - time cost
             scores (1D float array) - accuracy 
    Makes simple plot.
    Returns : 
    '''
    xs = np.array(times)
    ys = np.array(scores)

    plt.plot(xs, ys)
    return plt

def fit_get_time(pipe, x_train, y_train):
    '''
    Params : pipe (Pipeline) - pipeline
             x_train (1D array) - feature training set
             y_train (1D array) - label training set

    Fits pipeline, records the time interval.

    Returns : fit pipeline (1Darray),
              total time (int)
    '''
    start = time.time()
    fit_pipe = pipe.fit(X=x_train, y=y_train)
            
    stop = time.time() 
    time_total = stop - start
    return fit_pipe, time_total

def make_pipe_(sc, dim_red, kern, num_iter, 
               c, gam = 'scale', deg = 3):
    '''
    Params : sc (StandardScaler) - standardization method
             dim_red (PCA | LDA)- dimension reduction method
             kern ('linear' | 'rbf' | 'poly') - kernal
             num_iter (int) - max iterations
             C (float) - SVC regularization hyperparameter 
             Gamma (String | float) - 
    Utility function that makes a pipeline based
    on C, Gamma, and Degree. 

    Returns : Pipeline 

    '''
    pipe = Pipeline([("scaler",sc),
                    ("dim_red",dim_red),
                    ("model", SVC(C=c, gamma=gam, degree=deg, kernel=kern, max_iter=num_iter))],
                    verbose=True)
    return pipe
            

def svc_train(dim_red, kern, sc, data, iterations):
    '''
    params: dim_red - type of dimension reduction
            kern - kernal 
            sc - StandardScaler
            data - data[0] : X_train
                   data[1] : Y_train
                   data[2] : X_test
                   data[3] : Y_test

    Creates the various pipelines, fits, scores, and
    stores all the results in the results_ dictionary 

    Returns : dictionary storing: 
              "dim_red" : dim_red,
              "kernal" : kern,
              "num_iter" : [],
              "cs" : [],
              "gammas" : [],
              "degs" : [],
              "pipes" : [],
              "scores" : [],
              "times" : []
    '''
    if len(Cs) == len(GAMMAs) == len(DEGREEs):
        num_hyperparams = len(Cs)
    else:
        return None

    #seperate data into train and test sets
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]


    results_ = {
            "dim_red" : [],
            "kernal" : [],
            "num_iter" : [],
            "cs" : [],
            "gammas" : [],
            "degs" : [],
            "scores" : [],
            "pipes" : [],
            "times" : []
            }

    
    #make pipe -> fit -> get scores -> store results
    for iteration in iterations:   
        for j in range(num_hyperparams):
            c = Cs[j]
            gam = GAMMAs[j]
            deg = DEGREEs[j]
            if kern == "linear":
                pipe = make_pipe_(sc=sc, dim_red=dim_red, kern=kern, num_iter=iteration,
                                  c=c)
            elif kern == "rbf":
                pipe = make_pipe_(sc=sc, dim_red=dim_red, kern=kern, num_iter=iteration,
                                  c=c, gam=gam)
            else:   #poly
                pipe = make_pipe_(sc=sc, dim_red=dim_red, kern=kern, num_iter=iteration,
                                  c=c, gam=gam, deg=deg)

            fit_pipe, time_ = fit_get_time(pipe, x_train, y_train)

            score =  fit_pipe.score(x_test, y_test)

            results_['dim_red'].append(str(dim_red))
            results_['kern'].append(kern)
            results_['num_iter'].append(iteration)
            results_['cs'].append(c)
            results_['gammas'].append(gam)
            results_['degs'].append(deg)
            results_['pipes'].append(pipe)
            results_['scores'].append(score)
            results_['times'].append(time_)
    
    return results_


#%% run MNIST for all pca dim = 50
data = [X_train, Y_train, X_test, Y_test]
pca_results = []

for k in KERNALS:
    pca_50 =  PCA(n_components=50)
    results = svc_train(dim_red=pca_50,
                            sc=sc,
                            kern=k,
                            data=data,
                            iterations=ITERATIONS)
    pca_results.append(results)

#save all to pca50.csv
save_n_print_results(pca_results, FILE_PATHS['mnist']['pca50']) 
#save linear
#save rbf
#save poly

#%% run all MNIST
# 3.2 PCA50 vs PCA100 vs PCA200 LDA
lda = LinearDiscriminantAnalysis()

data = [X_train, Y_train, X_test, Y_test]

for k in KERNALS:
    for p in PCA_DIMS:
        pca = PCA(n_components=p)
        pca_results = svc_train(dim_red=pca,
                                sc=sc,
                                kern=k,
                                data=data,
                                iterations=ITERATIONS)
    
    lda_results = svc_train(dim_red=lda,
                            sc=sc,
                            kern=k,
                            data=data,
                            iterations=ITERATIONS)


#%% linear
