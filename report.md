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

Next, we explore the utility of principal component analysis (PCA) and linear discriminant analysis (LDA) in reducing the complexity of the data we are working with. PCA aims to find sets of coordinate axes (often rotated somewhat from a traditional Cartesian plane). LDA, on the other hand, projects its data down to a lower dimension to see if data can still be sufficiently explained. Dimensionality reduction allows us to produce models that are still expressive yet require less computational power to produce. 