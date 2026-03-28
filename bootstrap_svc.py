#Task 4 : 8(x3 kernels) SVC Voters vs 1(x3 kernels) SVC
from sklearn.model_selection import KFold

num_sets = 8

# TODO divide MNIST into 8 disjoint sets using Kfold
mnist_X_all = np.concatenate([X_train, X_test], axis=0)
mnist_Y_all = np.concatenate([Y_train, Y_test], axis=0)
kf = KFold(n_splits=num_sets, shuffle=True, random_state=SEED)

mnist_batches = {"x_trains" : [],
                 "y_trains" : [],
                 "x_tests" : [],
                 "y_tests" : []
                }
for train_index, test_index in kf.split(mnist_X_all):
    mnist_batches["x_trains"].append(mnist_X_all[train_index])
    mnist_batches["y_trains"].append(mnist_Y_all[train_index])
    mnist_batches["x_tests"].append(mnist_X_all[test_index])
    mnist_batches["y_tests"].append(mnist_Y_all[test_index])
    


fashion_X_all = np.concatenate([X_train_f, X_test_f], axis=0)
fashion_Y_all = np.concatenate([Y_train_f, Y_test_f], axis=0)
kf_f = KFold(n_splits=num_sets, shuffle=True, random_state=SEED)

fashion_batches = {"x_trains" : [],
                   "y_trains" : [],
                   "x_tests" : [],
                   "y_tests" : []
                  }


for f_train_index, f_test_index in kf_f.split(fashion_X_all):
    fashion_batches["x_trains"].append(fashion_X_all[f_train_index])
    fashion_batches["y_trains"].append(fashion_Y_all[f_train_index])
    fashion_batches["x_tests"].append(fashion_X_all[f_test_index])
    fashion_batches["y_tests"].append(fashion_Y_all[f_test_index])
    
    

'''mnist bootstrap'''
m_boot_pipes = []
for i in range(num_sets):
    data = [mnist_batches["x_trains"][i],
            mnist_batches["y_trains"][i],
            mnist_batches["x_tests"][i],
            mnist_batches["y_tests"][i],
           ]
    m_pipes_ = run_all_variations(data, 
                       folder_name='mnist', 
                       kerns=KERNALS, 
                       pca_comps=PCA_DIMS, 
                       num_iter=ITERATIONS, 
                       fpaths=FILE_PATHS)
    m_boot_pipes.extend(m_pipes_)

'''fashion bootstrap'''
f_boot_pipes = []
for i in range(num_sets):
    data = [fashion_batches["x_trains"][i],
            fashion_batches["y_trains"][i],
            fashion_batches["x_tests"][i],
            fashion_batches["y_tests"][i],
           ]
    f_pipes_ = run_all_variations(data, 
                       folder_name='fashion', 
                       kerns=KERNALS, 
                       pca_comps=PCA_DIMS, 
                       num_iter=ITERATIONS, 
                       fpaths=FILE_PATHS)
    f_boot_pipes.extend(f_pipes_)