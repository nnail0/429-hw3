import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist, fashion_mnist

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#ITERATIONS = [2000,5000,10000]
#Cs = [10, 1, 0.1, 0.01, 0.001, 0.0001]
#GAMMAs = ["scale", "auto", 0.1, 0.01, 0.001, 0.0001]

PCA_DIMS = [50,200]
#KERNALS = ['linear', 'rbf', 'poly']
KERNALS = ['rbf']
#ITERATIONS = [1000,5000,10000]
ITERATIONS = [3000]
#Cs = [100, 10.0, 5.0, 1.0, 0.1, 0.01,0.01, 0.001, 0.0001]
Cs = [2,2,1.5,1.5]
#GAMMAs = ["scale", "auto", 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.0001]
GAMMAs = [0.0005,0.005,0.005,0.001]
#DEGREEs = [1,1,2,2,2,3,4,4,5]
DEGREEs = [1,1,2,2]
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

standard_pl : Pipeline
fashion_pl : Pipeline
sc = StandardScaler()

def save_n_print_results(results, file_path):
    if isinstance(results, dict):
        results = [results]  
    df = pd.concat([pd.DataFrame(r) for r in results if r is not None], ignore_index=True)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
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
            

def svc_train(dim_red, kern, sc, data, iterations,fp, shuffle=False):
    '''
    params: dim_red - type of dimension reduction
            kern - kernal 
            sc - StandardScaler
            data - data[0] : X_train
                   data[1] : Y_train
                   data[2] : X_test
                   data[3] : Y_test
            fp - file-path

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
        raise ValueError(f"Cs, GAMMAs, DEGREEs must be same length. Got {len(Cs)}, {len(GAMMAs)}, {len(DEGREEs)}")

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
            "train_scores" : [],
            "test_scores" : [],
            "pipes" : [],
            "times" : []
            }

    print(f"Starting svc_train({str(dim_red)}{kern}{iterations})")
    #shuffle each of the global hyperparameter arrays to save time
    #instead of GridSearch
    #cs      = rand_gen.permutation(Cs)
    #gams    = rand_gen.permutation(GAMMAs)
    #degs    = rand_gen.permutation(DEGREEs)
    if shuffle:
        #no seed, want random to randomly shuffle the arrays 
        print("Permutating hyperparameters...")
        rand_gen = np.random.RandomState()
        idx_c   = rand_gen.permutation(len(Cs))
        idx_g   = rand_gen.permutation(len(GAMMAs))
        idx_d   = rand_gen.permutation(len(DEGREEs))

        cs   = [Cs[i]      for i in idx_c]
        gams = [GAMMAs[i]  for i in idx_g]
        degs = [DEGREEs[i] for i in idx_d]

    
        print(cs)
        print(gams)
        print(degs)
    cs = Cs
    gams = GAMMAs
    degs = DEGREEs

    
    #make pipe -> fit -> get scores -> store results
    for iteration in iterations:   
        for j in range(num_hyperparams):
            c = cs[j]
            gam = gams[j]
            deg = degs[j]
            print("---------------------------------------- Starting... ----------------------------------------")
            print(f"dim_red={str(dim_red)} | num_iter={iteration} | kernal={kern} | C={c} | gamma={gam} | degree={deg}")
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

            train_score =  fit_pipe.score(x_train, y_train)
            test_score =  fit_pipe.score(x_test, y_test)
            

            results_['dim_red'].append(str(dim_red))
            results_['kernal'].append(kern)
            results_['num_iter'].append(iteration)
            results_['cs'].append(c)
            results_['gammas'].append(gam)
            results_['degs'].append(deg)
            results_['pipes'].append(pipe)
            results_['train_scores'].append(train_score)
            results_['test_scores'].append(test_score)
            results_['times'].append(time_)
        
        
            print(f"dim_red={str(dim_red)} | num_iter={iteration} | kernal={kern} | C={c} | gamma={gam} | degree={deg} | time={time_} | train_score= {train_score} | test_score= {test_score}")
            print("---------------------------------------- Complete! ----------------------------------------\n\n")
        
        save_n_print_results(results_, fp)    
    
    return results_


'''FASHION ONLY'''
data = [X_train_f, Y_train_f, X_test_f, Y_test_f]
pca_lda_results = []

#TODO add score for trainging and check for overfitting
for k in KERNALS:
    pca_lda_results = []
    for p in PCA_DIMS:
        pca =  PCA(n_components=p)
        
        if p==50:
            results = svc_train(dim_red=pca,
                                    sc=sc,
                                    kern=k,
                                    data=data,
                                    iterations=ITERATIONS,
                                    fp=FILE_PATHS['fashion']['pca50'])
        elif p==100: 
            results = svc_train(dim_red=pca,
                                    sc=sc,
                                    kern=k,
                                    data=data,
                                    iterations=ITERATIONS,
                                    fp=FILE_PATHS['fashion']['pca100'])
        elif p==200: 
            results = svc_train(dim_red=pca,
                                    sc=sc,
                                    kern=k,
                                    data=data,
                                    iterations=ITERATIONS,
                                    fp=FILE_PATHS['fashion']['pca200'])
    lda = LinearDiscriminantAnalysis()
    lda_results = svc_train(dim_red=lda,
                            sc=sc,
                            kern=k,
                            data=data,
                            iterations=ITERATIONS,
                            fp=FILE_PATHS['fashion']['lda'])
        
    
    pca_lda_results.append(results)
    pca_lda_results.append(lda_results)
    if k=='linear': save_n_print_results(pca_lda_results,   #save linear
                         FILE_PATHS['fashion']['linear'])
    elif k=='rbf': save_n_print_results(pca_lda_results,    #save rbf
                         FILE_PATHS['fashion']['rbf'])
    elif k=='poly': save_n_print_results(pca_lda_results,   #save poly
                         FILE_PATHS['fashion']['poly'])
    
    
'''MNIST ONLY'''
data = [X_train, Y_train, X_test, Y_test]
pca_lda_results = []

for k in KERNALS:
    pca_lda_results = []
    for p in PCA_DIMS:
        pca =  PCA(n_components=p)

        if p==50:
            results = svc_train(dim_red=pca,
                                    sc=sc,
                                    kern=k,
                                    data=data,
                                    iterations=ITERATIONS,
                                    fp=FILE_PATHS['mnist']['pca50'])
        elif p==100: 
            results = svc_train(dim_red=pca,
                                    sc=sc,
                                    kern=k,
                                    data=data,
                                    iterations=ITERATIONS,
                                    fp=FILE_PATHS['mnist']['pca100'])
        elif p==200: 
            results = svc_train(dim_red=pca,
                                    sc=sc,
                                    kern=k,
                                    data=data,
                                    iterations=ITERATIONS,
                                    fp=FILE_PATHS['mnist']['pca200'])
    lda = LinearDiscriminantAnalysis()
    lda_results = svc_train(dim_red=lda,
                            sc=sc,
                            kern=k,
                            data=data,
                            iterations=ITERATIONS
                            fp=FILE_PATHS['mnist']['lda'])
        
    
    pca_lda_results.append(results)
    pca_lda_results.append(lda_results_)
    if k=='linear': save_n_print_results(pca_lda_results,   #save linear
                         FILE_PATHS['mnist']['linear'])
    elif k=='rbf': save_n_print_results(pca_lda_results,    #save rbf
                         FILE_PATHS['mnist']['rbf'])
    elif k=='poly': save_n_print_results(pca_lda_results,   #save poly
                         FILE_PATHS['mnist']['poly'])
    

#Task 4 : 8(x3 kernels) SVC Voters vs 1(x3 kernels) SVC
import sklearn.model_selection 
# TODO train set of 8 SVC models

# TODO divide MNIST into 8 disjoint sets

# TODO final prediction result obtained by voting

# MNIST
# 'linear'

# 'rbf'

# 'poly'

# Fashion
# 'linear'

# 'rbf'

# 'poly'

    
