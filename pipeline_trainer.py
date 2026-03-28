import numpy as np
import pandas as pd
import os
import time

from tensorflow.keras.datasets import mnist, fashion_mnist

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


class SVCPipelineTrainer(object):
    '''
        Params: folder_name (String) -  either "fashion" or "mnist"
                data - data[0] : X train
                       data[1] : Y train
                       data[2] : X test
                       data[3] : Y test
                kernals (1D int arr) - list of SVC kernals
                pca_dims (1D int arr) - list of n_components to pass to PCA
                num_iter (1D int arr) - list of max_iterations to run 
                fpaths (1D string arr) - list of file paths 
    '''
    def __init__(self, kerns, num_iter, pca_comps,
                 folder_name, data, fpaths, 
                 cs, gammas, degees):
        self.kerns = kerns
        self.num_iter = num_iter
        self.pca_comps = pca_comps
        self.folder_name = folder_name
        self.fpaths = fpaths
        self.data = data
        
        self.pipes = []
        self.rand_gen = np.random.RandomState()
        
        self.all_results = {"dataset" : [],
                            "scaler"  : [],
                            "dim_obj" : [],
                            "results" : []
                           }
        
    
    def save_n_print_results(self,results, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if isinstance(results, dict):
            results = [results]  
        df = pd.concat([pd.DataFrame(r) for r in results if r is not None], ignore_index=True)
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        print(df.to_string(index=False))
    '''
    def plot_t_vs_acc(times, scores, num_iter):
        
        Params : times (1D float array) - time cost
                 scores (1D float array) - accuracy 
        Makes simple plot.
        Returns : 
        
        xs = np.array(times)
        ys = np.array(scores)
    
        plt.plot(xs, ys)
        return plt
     '''   
    
    def fit_get_time(self,pipe, x_train, y_train):
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
    
    def make_pipe_(self,dim_red, kern, num_iter,sc, 
                   c, gam = 'scale', deg = 3):
        '''
        Params : 
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
                
    
    def svc_train(self,dim_red, kern, sc,fp, shuffle=True):
        '''
        params: dim_red - type of dimension reduction
                kern - kernal 
                sc - StandardScaler
                fp - file-path
                shuffle - shuffle the cs_, gammas_, and degrees_
    
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
        if len(self.cs_) == len(self.gammas_) == len(self.degrees_):
            num_hyperparams = len(self.cs_)
        else:
            raise ValueError(f"Cs, GAMMAs, DEGREEs must be same length. Got {len(self.cs_)}, {len(self.gammas_)}, {len(self.degrees_)}")

        results = {
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
    
        #seperate data into train and test sets
        x_train = self.data[0]
        y_train = self.data[1]
        x_test = self.data[2]
        y_test = self.data[3]
    

    
        print(f"Starting svc_train({str(dim_red)}{kern}{iterations})")
        #shuffle each of the global hyperparameter arrays to save time
        #instead of GridSearch
        if shuffle:
            #no seed, want random to randomly shuffle the arrays 
            print("Permutating hyperparameters...")
            
            idx_c   = self.rand_gen.permutation(len(self.cs_))
            idx_g   = self.rand_gen.permutation(len(self.gammas_))
            idx_d   = self.rand_gen.permutation(len(self.degrees_))
    
            self.cs_   = [self.cs_[i]      for i in idx_c]
            self.gammas_ = [self.gammas_[i]  for i in idx_g]
            self.degrees_ = [self.degrees_[i] for i in idx_d]    
    
        
        #make pipe -> fit -> get scores -> store results
        for iteration in self.num_iter:   
            for j in range(num_hyperparams):
                c = self.cs_[j]
                gam = self.gammas_[j]
                deg = self.degrees_[j]
                print("---------------------------------------- Starting... ----------------------------------------")
                print(f"dim_red={str(dim_red)} | num_iter={iteration} | kernal={kern} | C={c} | gamma={gam} | degree={deg}")
                if kern == "linear":
                    pipe = self.make_pipe_(sc=sc, dim_red=dim_red, kern=kern, num_iter=iteration,
                                      c=c)
                elif kern == "rbf":
                    pipe = self.make_pipe_(sc=sc, dim_red=dim_red, kern=kern, num_iter=iteration,
                                      c=c, gam=gam)
                else:   #poly
                    pipe = self.make_pipe_(sc=sc, dim_red=dim_red, kern=kern, num_iter=iteration,
                                      c=c, gam=gam, deg=deg)
    
                fit_pipe, time_ = fit_get_time(pipe, x_train, y_train)
    
                train_score =  fit_pipe.score(x_train, y_train)
                test_score =  fit_pipe.score(x_test, y_test)
                
    
                results['dim_red'].append(str(dim_red))
                results['kernal'].append(kern)
                results['num_iter'].append(iteration)
                results['cs'].append(c)
                results['gammas'].append(gam)
                results['degs'].append(deg)
                results['pipes'].append(pipe)
                results['train_scores'].append(train_score)
                results['test_scores'].append(test_score)
                results['times'].append(time_)
            
            
                print(f"dim_red={str(dim_red)} | num_iter={iteration} | kernal={kern} | C={c} | gamma={gam} | degree={deg} | time={time_} | train_score= {train_score} | test_score= {test_score}")
                print("---------------------------------------- Complete! ----------------------------------------\n\n")
            
            save_n_print_results(self.results, fp)    
        
        return results
    
    
    
    def run_all_variations(self):
        '''
        '''

        if self.folder_name != 'fashion' and self.folder_name != 'mnist':
            raise ValueError(f'Expected either folder_name fashion or mnist but got: {self.folder_name}')
        
        
        
        for k in self.kerns:                       # run for all kernal types
            pca_lda_results = []
            for p in self.pca_comps:               # run for all PCA dimensions 
                pca =  PCA(n_components=p)
                sc = StandardScaler()
                
                if p==50:
                    results = self.svc_train(dim_red=pca,
                                            sc=sc,
                                            kern=k,
                                            fp=self.fpaths[f'{self.folder_name}']['pca50'])
                elif p==100: 
                    results = self.svc_train(dim_red=pca,
                                            sc=sc,
                                            kern=k,
                                            fp=self.fpaths[f'{self.folder_name}']['pca100'])
                elif p==200: 
                    results = self.svc_train(dim_red=pca,
                                            sc=sc,
                                            kern=k,
                                            fp=self.fpaths[f'{self.folder_name}']['pca200'])
                    
                pca_lda_results.append(results)
                #self.pipes.extend(results['pipes'])
                
            # run LDA once for every 3xPCA
            sc = StandardScaler()
            lda = LinearDiscriminantAnalysis()
            lda_results = self.svc_train(dim_red=lda,
                                    sc=sc,
                                    kern=k,
                                    fp=self.fpaths[f'{self.folder_name}']['lda'])
            
            #self.pipes.extend(lda_results['pipes'])
            pca_lda_results.append(lda_results)
            
            if k=='linear': self.save_n_print_results(pca_lda_results,   #save linear
                                 fpaths[f'{self.folder_name}']['linear'])
            elif k=='rbf': self.save_n_print_results(pca_lda_results,    #save rbf
                                 fpaths[f'{self.folder_name}']['rbf'])
            elif k=='poly': self.save_n_print_results(pca_lda_results,   #save poly
                                 fpaths[f'{self.folder_name}']['poly'])
        return pca_lda_results
        
        
        
    