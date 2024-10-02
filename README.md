# Economy-and-Population-Growth

This research project is being carried out by Geroge Kontos, an Honours Student of the Honours Programme Bachelor of TU Delft. The research is being conducted under the supervision of Dr. N. Parolya.

## Project Description

This project aims to investigate the effectiveness of machine learning approaches for long-horizon forecasting of international financial growth. This project uses the predictions published by Dr. U.K. Mueller, Dr. J.H.Stock and Dr. Mark. W. Watson in their work **An Econometric Model of International Growth Dynamics for Long-horizon Forecasting** as a baseline for evaluation and comparisons of the foundings of machine learning models.

To the end of providing high quality predictions, two types of machine learning algorithms are being used; clustering algorithms and regressive neural network. Clustering allows grouping the GDP growth of different countries that present similar characteristics together. Afterwards, regressive neural networks can be tuned to perform iterative predictions on the GDP data of each country. Tuning one neural network per cluster significantly increases the computational effeciency.

## Project Sections

### Replication of Previous Work

The [SCC_Replication](/SCC_Replication) directory contains a Python implementation of the Gibbs Sampler described by Ulrich K. Muller, James H. Stock and Mark W. Watson in their work **An Econometric Model of International Growth Dynamics for Long-horizon Forecasting**,as well as a documentation of the previous Fortran90 implementation. A Markov Chain Monte Carlo algorithm is used to estimate the distribution of specific parameters of the cross-country linear econometric model they describe.

### Neural Networks

The [NeuralNetworks](/NeuralNetworks) directory contains all the necessary modules for performing iterative predictions with Multi-Layer Perceptrons and Recurrent Neural Networks. Specifically:
- [LearningInstance.py](/NeuralNetworks/LearningInstance.py): A data structure that contains all the required data for training and testing a neural network for a specified country
- [PreProcessing.py](/NeuralNetworks/PreProcessing.py): All the data pre-processing functions required to construct a `LearningInstance` object for each country
- [ConstructModels.py](/NeuralNetworks/ConstructModels.py): Functions for constructing neural networks of selected specifications. These functions are required for automating the Bayesian Optimization tuning of the neural networks
- [FineTuning.py](/NeuralNetworks/FineTuning.py): Functions that tune the number of layers and the number of neurons per layer for the neural networks. Tuning is performed through Bayesian Optimization
- [Predictions.py](/NeuralNetworks/Predictions.py): Functions that perform iterative predictions for the different types of neural networks
- [PostProcessing.py](/NeuralNetworks/PostProcessing.py): Functions for visualizing the testing and predictions

### Clustering

The [Clustering](/Clustering) directory contains all the necessary modules for clustering the countries using hierarchical, spectral and partitional clustering. Specifically:
- [PreProcessing.py](/Clustering/PreProcessing.py): The pre-processing functions required to construct the final dataset
- [Outliers.py](/Clustering/Outliers.py): Functions for outlier detection through clustering, using the DBSCAN algorithm
- [HierarchicalClustering.py](/Clustering/HierarchicalClustering.py): Functionality for constracting tree structures for hierarchical clustering using different linkage methods and cutting the hierarchy tree to derive the optimal number of clusters
- [SpectralClustering.py](/Clustering/SpectralClustering.py): Functionality for tuning and implementing the different spectral clustering algorithms
- [TimeSeriesPartitioning.py](/Clustering/TimeSeriesPartitions.py): Functionality for implementing partitioning algorithms specifically designed for time series clustering
- [hierarchical_clustering.ipynb](/hierarchical_clustering.ipynb): Provides a detailed process for determining the optimal hierarchical clustering algorithm
- [spectral_clustering.ipynb](/spectral_clustering.ipynb): Provides a detailed process for examining the effectiveness of spectral clustering for the given task
- [timeseries_clustering](/partitional_clustering.ipynb): Provides a detailed process for finding the best partioning approaches to time series clustering.

### Utilities

- [DataUtils.py](/Utils/DataUtils.py): Utility functions for reading and writing data to files.
- [VisualUtils.py](/Utils/VisualUtils.py): Utility functions for visualizing results