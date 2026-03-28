# CS 429 Midterm - Pipelining, Hyperparameter Tuning, Ensenble Learning

Nathan Nail, Miles Nordwall

Spring 2026

The midterm project asks us to use the MNIST and Fashion MNIST data using the `sklearn` pipeline to examine more about scaling, hyperparameter tuning, dimensionality reduction, and ensemble learning. 

### Background

To make examination and analysis of our results easier, we import the scikit-learn Pipeline as detailed in the project guidelines. This allows for a streamlined data processing experience in which data can be transformed in one of multiple different ways before being sent for model fit. 

The first step in this process was to use the scikit-learn `StandardScaler` to standardize the input matrix. By standardizing the data, we make the dimensionality reduction in the next step less computationally expensive. This will be the first step in our pipeline. 

```
sc = StandardScaler()
pipe = Pipeline(sc, ...)
```

We also explore the utility of principal component analysis (PCA) and linear discriminant analysis (LDA) in reducing the complexity of the data we are working with. PCA aims to find sets of coordinate axes (often rotated somewhat from a traditional Cartesian plane) that form "components", or groups of observations.. LDA, on the other hand, projects its data down to a lower dimension to see if data can still be sufficiently explained. Dimensionality reduction allows us to produce models that are still expressive yet require less computational power to produce. 

The model evaluation process was broken down into several steps:
1. Run several permutations of each kernel with each type of dimensionality reduction. Score these models based on their testing accuracy.
2. Find the models with the best testing accuracy and run these models again, checking for both testing *and* training accuracy. This allows for a check on overfit or underfit, which can especially be a problem with a nonlinear kernel. 

The utility of these models will be based upon their ability to produce accurate results within a specific timeframe, assuming they fit the data well. In many scenarios, it may be worthwhile to drop a few decimals, or maybe even a percentage point or two, to get a model with 99% of the expressiveness but higher efficiency. 

### General Model Training

Results are presented in order of kernel. Hyperparameters were tuned and carried over to new PCA/LDA selections/kernels to serve as a baseline. 

#### Linear

| Dim. Reduction Method 	| C     	| max_iter (default 1000) 	| Testing Score 	| Time  	|
|-----------------------	|-------	|-------------------------	|---------------	|-------	|
| PCA 50                	| 0.01  	|                         	| 0.9065        	| 28s   	|
| PCA 50                	| 0.001 	|                         	| 0.843         	| 39.0s 	|
| PCA 50                	| 0.1   	|                         	| 0.7014        	| 24.2s 	|
| PCA 100               	| 0.001 	|                         	| 0.9147        	| 50.8s 	|
| PCA 100               	| 0.01  	|                         	| 0.9346        	| 35.4s 	|
| PCA 100               	| 0.1   	|                         	| 0.7253        	| 34.5s 	|
| PCA 200               	| 0.01  	|                         	| 0.9366        	| 43.7s 	|
| PCA 200               	| 0.001 	| 1000 (did not converge) 	| 0.9407        	| 67.0  	|
| PCA 200               	| 0.001 	| 2000 (did not converge) 	| 0.942         	| 65.1  	|
| LDA                   	| 0.01  	| 1000 (did not converge) 	| 0.872         	| 18.3  	|
| LDA                   	| 0.01  	| 2000                    	| 0.8927        	| 19.0  	|
|                       	|       	|                         	|               	|       	|

#### RBF

In hindsight, gamma values as high as 10 may not be the best; theoretically, this would lead to a higher amount of overfitting. 

| Dim. Reduction Method  	| C     	| Gamma 	| max_iter (default 1000) 	| Testing Score 	| Time  	|
|------------------------	|-------	|-------	|-------------------------	|---------------	|-------	|
| PCA 50                 	| 0.01  	| 10    	| 1000 (did not converge) 	| 0.6109        	| 1.63m 	|
| PCA 50                 	| 0.1   	| 10    	| 2000                    	| 0.6615        	| 3.3m  	|
| PCA 50                 	| 0.01  	| 10    	| 1000                    	| 0.6616        	| 1.5m  	|
| PCA 50                 	| 0.001 	| 10    	| 1000                    	| 0.6087        	| 1.5m  	|
| PCA 50                 	| 1     	| 10    	| 1000                    	| 0.6132        	| 1.6m  	|
| PCA 50                 	| 0.1   	| 0.1   	| 1000 (did not converge) 	| 0.852         	| 1.5m  	|
| PCA 50                 	| 1     	| 0.1   	| 2000                    	| 0.8262        	| 2.9m  	|
| PCA 50                 	| 0.1   	| 0.01  	| 1000                    	| 0.9383        	| 1.4m  	|
| PCA 50                 	| 0.1   	| 0.001 	| 1000                    	| 0.8725        	| 1.1m  	|
| PCA 100                	| 0.1   	| 0.01  	| 1000 (did not converge) 	| 0.9343        	| 2.0m  	|
| PCA 100                	| 0.1   	| 0.01  	| 2000 (did not converge) 	| 0.888         	| 3.7m  	|
| PCA 100                	| 0.1   	| 0.001 	| 1000 (did not converge) 	| 0.8911        	| 1.6m  	|
| PCA 100                	| 0.01  	| 0.001 	| 1000 (did not converge) 	| 0.7002        	| 2.1m  	|
| PCA 200                	| 0.1   	| 0.01  	| 1000 (did not converge) 	| 0.9269        	| 2.7m  	|
| PCA 200                	| 0.1   	| 0.01  	| 2000 (did not converge) 	| 0.8739        	| 5.1m  	|


#### Polynomial

Given the shape of the input data, training mainly focused on degrees 3 through 5. Higher degrees on average led to more time required to train. 

| Dim. Reduction Used 	| Degree 	| C    	| Gamma 	| max_iter               	| Testing Score 	| Time  	|
|---------------------	|--------	|------	|-------	|------------------------	|---------------	|-------	|
| PCA 50              	| 3      	| 0.01 	| 5     	| 5000 (did not converge) 	| 0.9688        	| 10.9s 	|
| PCA 50              	| 3      	| 1    	| 10    	| 20000                  	| 0.9688        	| 12.8s 	|
| PCA 100             	| 3      	| 0.01 	| 10    	| 20000                  	| 0.9749        	| 28.8s 	|
| PCA 100             	| 3      	| 0.1  	| 10    	| 20000                  	| 0.9749        	| 33.1s 	|
| PCA 200             	| 3      	| 0.01 	| 10    	| 20000                  	| 0.9775        	| 1.3m  	|
| PCA 50              	| 4      	| 0.01 	| 5     	| 20000                  	| 0.9623        	| 15.5s 	|
| PCA 50              	| 4      	| 0.01 	| 0.1   	| 20000                  	| 0.9623        	| 17.6s 	|
| PCA 100             	| 4      	| 1    	| 5     	| 20000                  	| 0.9679        	| 43.1s 	|
| PCA 200             	| 4      	| 0.01 	| 5     	| 20000                  	| 0.9693        	| 1.9m  	|
| PCA 200             	| 4      	| 1    	| 5     	| 20000                  	| 0.9693        	| 1.9m  	|
| PCA 50              	| 5      	| 0.01 	| 0.1   	| 20000                  	| 0.9632        	| 56.1s 	|
| PCA 100             	| 5      	| 0.01 	| 5     	| 20000                  	| 0.9675        	| 1.2m  	|
| PCA 200             	| 5      	| 0.01 	| 5     	| 20000                  	| 0.968         	| 2.4m  	|


### Best Models + Overfit Check

Out of these models, several of the most accurate on the test data were selected for use. 

| Kernel + Dim. Reduction 	| C    	| Gamma 	| Deg. 	| Time  	|
|-------------------------	|------	|-------	|------	|-------	|
| Linear PCA 100          	| 0.01 	|       	|      	| 35.4s 	|
| Linear PCA 200          	| 0.01 	|       	|      	| 1.2m  	|
| Poly PCA 200 (Overfit)  	| 0.01 	| Any   	| 3    	| 2.0m  	|
| RBF PCA 50              	| 0.1  	| 0.01  	|      	| 1.2m  	|

There were others that had similar performance, but given that some of these iterations had a high `max_iter` value (even if a convergence warning printed), some models saw some overfitting. 

### Hyperparameter Tweaking
The best hyperparameters to use tended to vary with the kernel. Within kernels, it was often best to transfer over the previous best parameters to use as a baseline. Starting with a linear kernel and 50-component PCA, keeping C small (less weight on the error calculation) proved to provide good fit without underfitting. A small gamma was  beneficial for RBF and polynomial models, but RBF in particular benefitted from a larger value for C. For polynomial models, we did not need to go too far with the degree to see promising results. Going too high in degree (5, or sometimes even 4) had the tendency to result in overfitting. Dropping C back down to 1 or 0.1 along with a low gamma proved to be the most beneficial set for the polynomial runs. 

For the nonlinear kernels, the interplay between C and gamma was interesting. If gamma was too high, individual points could have a lot of influence on how the model is trained. However, tweaking the value of C can help provide some balance; higher values could punish misclassification. 

### PCA v. LDA: Analysis

When examining PCA vs. LDA, it seems that PCA was able to consistently outperform LDA in most cases, despite taking a bit longer. While LDA may seem to be a desirable alternative for its efficiency, its inabibility to cross even the 90% threshold in most cases makes it difficult to recommend. It would be worth the extra time to let PCA run in order to achieve scores above 95% on some kernels. 


Regardless of the kernel, LDA managed testing accuracies within the same ballpark on a consistent basis. Training on the fashion dataset, LDA managed around 84%, which allowed it to outperform the linear kernel in a shorter time period. However, this meant that it lagged behind PCA notably in nonlinear cases. 

The main standout feature of LDA is how little time it takes for it to train. For data that can be distinguished well with LDA, it may be a good candidate if one could manage to bump up its accuracy. If LDA were able to capture slightly more information with similar timing, it could be more viable as it allowed the SVC to train in less time. There may be some other applications aside from SVC pipelining where LDA may be a more viable option. 

### Kernel Comparisons (using PCA)

When considering different kernels, we desire a high accuracy but not if the extra time or computational cost of training is not worth it. It may be more suitable to use a kernel that runs much faster and leave a few tenths or hundredths of a percent on the table. Each kernel's strengths are broken down in more detail below: 

- The linear kernel was able to achieve an acceptable test accuracy without overfitting within a decent period of time on MNIST. For fashion, it seemed to lag behind in testin accuracy. It is a solid candidate for simpler data if efficiency is desired, given that it usually ran faster than the others 

- One of our members initally testes RBF, but was not super experimental with the parameters, initially about to recommend against. When viewing the data from the other group member, higher values of C with low values of gamma were employed, which provided great results. However, it is worth noting that some of the better RBF models took longer to train on the fashion dataset. RBF seems to deliver high accuracy on both simple and complex data, but the training time can vary and may not scale as well with more complex data. 

- One member's testing with the polynomial kernel led to promising results, but some of these were discovered to be overfitting later on. However, with the right hyperparameter combinations, polynomial kernels can provide satisfactory training accuracy without overfitting. With both simple and complex data, the polynomial kernel was able to provide accuracy on par with that of RBF while taking slightly less time to train. 

It seems that each kernel has differing strengths. While the linear kernel was never the most accurate, it was able to deliver sufficient results despite being the simplest of the models. This gives it a lower training time. Both RBF and polynomial kernels can be accurate on more complex data if one is willing to wait a little longer for training. For the sake of efficiency, the polynomial kernel may be a good fit, but there are some cases where the training times are a wash. In this case, the choice between the kernels will likely depend on the nature of the data (i.e. using a specific polynomial degree if the shape of the data in a feature space is known ahead of time). 

### Ensemble Learning with Bootstrap Aggregation
To replicate ensemble learning without using existing library code, we can train an arbitrary number of the same SVC, make each SVC predict on a sample, and use majority voting (one of multiple voting schemes) to determine the best label for specific input. In combination with this, we can use bootstrap aggregation (a.k.a. "bagging"). This involves selecting features for each SVC by sampling from the data with replacement. For more complex models, bagging can help prevent overfitting and reduce model variance. 
We can compare the testing accuracy and timing between single-model and bagging to see if there is an impact on the final results. 
