#Task 4 : 8(x3 kernels) SVC Voters vs 1(x3 kernels) SVC
from sklearn.model_selection import KFold
from collections import Counter

num_sets = 8

# TODO divide MNIST into 8 disjoint sets using Kfold
mnist_X_all = np.concatenate([X_train, X_test], axis=0) # merge test and train data
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
    
    
# TODO final prediction result obtained by voting
# take all y values and perform a majority vote for each sample. 
# return an ndarray of the same length. 
# training labels: 2D array
# pass in "X_trains"
def majority_vote(training_labels):
    results = []
    # 
    for i in range(len(training_labels)):
        curr_row = training_labels[i]
        print(curr_row)
        freq = Counter()
        for j in curr_row:
            freq[j] += 1
        result = freq.most_common(1)[0][0]
        print("Most common: ", result)
        results.append(result)
    
    return results

# all_mnist_train_results = np.stack(mnist_batches["y_trains"], axis=1)
# mnist_vote_result = majority_vote(all_mnist_train_results)

n_iter = 20000

p1 = Pipeline([("scaler", StandardScaler()), ("dim_red" , PCA(200)), ("model" , SVC(kernel='linear', C=0.001, max_iter=n_iter))], verbose=True)
p2 = Pipeline([("scaler", StandardScaler()), ("dim_red" , PCA(100)), ("model", SVC(kernel='rbf', C = 4, gamma = 0.001, max_iter=n_iter))], verbose=True)
p3 = Pipeline([("scaler", StandardScaler()), ("dim_red",  PCA(200)), ("model", SVC(kernel='poly', C = 0.1, gamma = 0.01, degree=2, max_iter=n_iter))], verbose=True)

pipes = [p1, p2, p3]

def run_pipe(pipeline : Pipeline, train_x, train_y, test_x, test_y) -> list:
    start_time = time.time()
    pipeline.fit(X = train_x, y = train_y)
    time_taken = time.time() - start_time

    test_score = pipeline.score(X = test_x, y = test_y)

    print("time: ", time_taken)
    print("test score: ", test_score)
    pipe_result = [test_score, time_taken]
    return pipe_result