# Machine Learning Approaches for Clustering and Forecasting GDP Growth

This research project is being carried out by Geroge Kontos, an Honours Student of the Honours Programme Bachelor of TU Delft. The research is being conducted under the supervision of Dr. N. Parolya.

## Abstract

This project aims to investigate the effectiveness of machine learning approaches for long-horizon forecasting of international financial growth. This project uses the predictions published by Dr. U.K. Mueller, Dr. J.H.Stock and Dr. Mark. W. Watson in their work [An Econometric Model of International Growth Dynamics for Long-horizon Forecasting](https://direct.mit.edu/rest/article/104/5/857/97738/An-Econometric-Model-of-International-Growth) as a baseline for evaluation and comparisons of the foundings of machine learning models.

## Process and Findings

### Clustering

In order to find a suitable way to cluster the evolution of annual GDP per capita of countries, the three most popular clustering approaches were examined: hierarchical clustering, spectral clustering and partional clustering. For hierarchical clustering and majority of the experiments with spectral clustering, the data set was augmented with population and bilateral currency exchange rate (against the US Dollar) per annum for each country, in an attempt to find out whether such information can provide any major insights.

#### Clustering-Based Outlier Detection

For hierarchical algorithms, DBSCAN was employed to perform clustering-based outlier detection. The parameters of the algorithm were determined by common heuristics. The complete implementation of the algorithm can be found in [Outliers.py](/Clustering/Outliers.py).

#### (Explicit) Centroid Calculation

Clustering in this problem is ultimately used to reduce the number of neural networks that must be tuned. However, in order to train one neural network per cluster, it is necessary to derive some common ground for all members of the clusters. While centroids are implicitly calculated and outputed by partitional algorithms on the GDP data, for algorithms like Kernel $k$-means or hierarchical clustering such cluster centers are not available. Inspired by the time series-specific implementation of the $k$-means algorithm, we explicitly calculate centroids whenever necessary using the [DTW (i.e. Dynamic Time Warping) Barycenter Averaging (DBA)](https://www.sciencedirect.com/science/article/abs/pii/S003132031000453X) algorithm. The selected configuration of the DBA algorithm is located in [TimeSeriesUtils.py](/Utils/TimeSeriesUtils.py).

#### [Hierarchical Clustering](/hierarchical_clustering.ipynb)

Hierarchical clustering was performed on the augmented dataset, considering GDP per capita, population, and currency exchange rate per anum for each country, from 1960 until 2017. For the clustering algorithm to be more reasonable, as clustering on $174$ features would be unconventional, dimensionality reduction by Principal Component Analysis (PCA) is used. After the dataset is reduced, DBSCAN was used to detect outliers; in this case, China, India, Indonesia, Iran and Vietnam were removed from the dataset.

Four different linkage methods were tested for this algorithm; complete, average, single and ward linkage. Ward linkage turned out to be the best suited for this problem, producing the most balanced dendogram and the highest silhouette score performance. 

![Ward Linkage Dendrogram](/Images/Clustering/Hierachical/ward_linkage.png)

However, even though the dendogram suggests clear clusters, this is not translated successfully to the GDP time series. While some of the data demonstrates intra-cluster similarities, the different clusters are completely undistinguishable visually.

![Hiearchical Clustering](/Images/Clustering/Hierachical/hierarchical_clusters_plot.png)

Furthermore, outlier detection performed on the PCA-reduced dataset is not effective in detecting outlier time series. On the contrary, some of the outliers seem to have very similar structure and magnitude with some of the centroids.

![Hierarchical Clustering Centroids and Outliers](/Images/Clustering/Hierachical/hier_centroids_outliers.png)

#### [Spectral Clustering](/spectral_clustering.ipynb)

Spectral Clustering was performed in both the original and augmented dataset, with different configurations. The number of clusters are determined by the 'Eigengap' Heuristic suggested by literature.

![Eigenvalues Plot with Eigengap](/Images/Clustering/Spectral/eigengap.png)

Three different types of similarity matrices where used: $k$-neighbors graph  and $\epsilon$-neighborhood with Euclidean Distance Measure, as well as a $k$-neighbors graph with DTW as a dissimilarity measure. The versions using Euclidean Distance suggest heavy imbalance of the clusters; in fact, the $\epsilon$-neighborhood version splits the data in 19 different clusters, with 18 of them having no more than 6 countries each, and one of them having 66 countries. Even though the version using DTW provides the most balanced result and achieves the best silhouette score of the three algorithms, its clusters lack withing group similarity. This becomes even clearer when observing the (explicitely calculated) centroids.

#### [Partitional Clustering](/partitional_clustering.ipynb)

Literature has shown that several popular partitional algorithms can be adjusted to work better with time series by leveraging the concept of [dynamic time warping](https://rtavenar.github.io/blog/dtw.html) and all the related structures and algorithms (DTW Barycenter Averaging, Global Alignment Kernels, etc.). In this work, five such algorithms are tested: traditional $k$-Means (using Euclidean Distance), $k$-Means using DTW distance, [k-Shapes](https://sigmodrecord.org/publications/sigmodRecord/1603/pdfs/18_kShape_RH_Paparrizos.pdf), [k-Medoids](https://wis.kuleuven.be/stat/robust/papers/publications-1987/kaufmanrousseeuw-clusteringbymedoids-l1norm-1987.pdf) using DTW distance and [kernel k-Means](https://www.cs.utexas.edu/~inderjit/public_papers/kdd_spectral_kernelkmeans.pdf) using the Global Alignment Kernel (GAK) as a kernel function. In order to determine which algorithm is optimal and with which number of clusters, the elbow heuristic was used. Only the annual GDP per capita data was considered.

![Elbow Plot](/Images/Clustering/Partitional/elbow_plot.png)

Kernel $6$-Means using GAK performs the best, very well structured clusters. Furthermore, utilizing spectral clustering for the initialization of the algorithm further improves these findings. For each of the two algorithms, the clusters and their visualization on the world map are shown below. Countries that are not included in the dataset are also omitted from the map. Generally, all 5 partitional algorithms, when used with 4-6 clusters, significantly outperform hierarchical and spectral approaches in terms of silhouette score and cluster cohesion. 

![Kernel 6-Means with GAK](/Images/Clustering/Partitional/kernel_kmeans_plots.png)
![Kernel 6-Means with GAK (map)](/Images/Clustering/Partitional/kernel_kmeans_map.png)

![Kernel 6-Means with GAK and spectral initialization](/Images/Clustering/Partitional/kernel_kmeans_spec_plots.png)
![Kernel 6-Means with GAK and spectral initialization (map)](/Images/Clustering/Partitional/kernel_kmeans_spec_map.png)

The following graph displays an example of information for countries that belong to groups that arise from geography (e.g. Europe), alliances (e.g. Former Soviet Block), the spoken language (e.g. Anglo-Saxon countries) or other characteristics (e.g. Island nations) and how this is reflected by the clustering algorithms. The weight of the edges is the DTW distance of their GDP time series, the color of the edge indicates whether they belong to the same cluster and the thickness indicates their geographical distance. 

![Former Soviet Block](/Images/Clustering/Partitional/soviet_block.png)

#### [Clustering on Non-zero mean data](/unscaled_clustering.ipynb)

In order to capture the magnitude difference of different countries, the same experiment is performed with a reduced pre-processing pipeline: The logarithm of each time series is adjusted to have unit variance, but its mean remains unchanged. The best performing algorithm in this case was Kernel $13$-Means, providing very well structured clusters and capturing geographical relations better than any other algorithm. The results are indicated by the two following figures.

![Kernel 13-Means with GAK and non-zero mean input](/Images/Clustering/Partitional/kernel_13plots.png)
![Kernel 13-Means with GAK and non-zero mean input (map)](/Images/Clustering/Partitional/kernel_13map.png)

#### Conclusion

Based on these results, we can conclude that the order of performance of the three approaches is clear; hierarchical clustering performs the worst, while partitional clustering performs the best. For the remainder of this project, the $4$-Means algorithm using DTW will be preferred over the $4$-Medoids version, because $k$-Medoids only considers input data as possible centroids, which, while robust to outliers, might not be optimal in representing the majority of the data within a cluster.

## Directories and Modules

### Home Directory

Jupyter notebooks in the home directory document different algorithms and experiments followed in this project. Specifically:
- [hierarchical_clustering.ipynb](/hierarchical_clustering.ipynb): Provides a detailed process for determining the optimal hierarchical clustering algorithm
- [neural_networks.ipynb](/neuralnetworks.ipynb): Provides a detailed explenation of how neuralnetworks are tuned and used for iterative forecasting
- [partitional_clustering.ipynb](/partitional_clustering.ipynb): Provides a detailed process for finding the best partioning approaches to time series clustering.
- [prob_forecasting.ipynb](/prob_forecasting.ipynb): Performs probabilistic forecasting on the GDP growth
- [spectral_clustering.ipynb](/spectral_clustering.ipynb): Provides a detailed process for examining the effectiveness of spectral clustering for the given task.
- [unscaled_clustring.ipynb](/unscaled_clustering.ipynb): Performs partitional clustering on the non-zero mean time series

### Replication of Previous Work

The [SCC_Replication](/SCC_Replication) directory contains a Python implementation of the Gibbs Sampler described by Ulrich K. Muller, James H. Stock and Mark W. Watson in their work [An Econometric Model of International Growth Dynamics for Long-horizon Forecasting](https://direct.mit.edu/rest/article/104/5/857/97738/An-Econometric-Model-of-International-Growth),as well as a documentation of the previous Fortran90 implementation. A Markov Chain Monte Carlo algorithm is used to estimate the distribution of specific parameters of the cross-country linear econometric model they describe.

### Clustering

The [Clustering](/Clustering) directory contains all the necessary modules for clustering the countries using hierarchical, spectral and partitional clustering. Specifically:
- [HierarchicalClustering.py](/Clustering/HierarchicalClustering.py): Functionality for constracting tree structures for hierarchical clustering using different linkage methods and cutting the hierarchy tree to derive the optimal number of clusters
- [KernelKMeans.py](/Clustering/KernelKMeans.py): Adaptation of [tslearn](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html#tslearn.clustering.KernelKMeans)'s implementation of `KernelKMeans` to allow nonrandom initialization.
- [Outliers.py](/Clustering/Outliers.py): Functions for outlier detection through clustering, using the DBSCAN algorithm
- [SpectralClustering.py](/Clustering/SpectralClustering.py): Functionality for tuning and implementing the different spectral clustering algorithms
- [TimeSeriesPartitioning.py](/Clustering/TimeSeriesPartitions.py): Functionality for implementing partitioning algorithms specifically designed for time series clustering

### Forecasting
The [Forecasting](/Forecasting/) directory contains all the nesessary functionality for performing univariate and multivariate probabilistic forecasts. Specifically:
- [UnivariateForecasts.py](/Forecasting/UnivariateForecasts.py): Functions for univariate recursive probabilistic forecasts.

### Neural Networks

The [NeuralNetworks](/NeuralNetworks) directory contains all the necessary modules for performing iterative predictions with Multi-Layer Perceptrons and Recurrent Neural Networks. Specifically:
- [ConstructModels.py](/NeuralNetworks/ConstructModels.py): Functions for constructing neural networks of selected specifications. These functions are required for automating the Bayesian Optimization tuning of the neural networks
- [FineTuning.py](/NeuralNetworks/FineTuning.py): Functions that tune the number of layers and the number of neurons per layer for the neural networks. Tuning is performed through Bayesian Optimization
- [LearningInstance.py](/NeuralNetworks/LearningInstance.py): A data structure that contains all the required data for training and testing a neural network for a specified country
- [PostProcessing.py](/NeuralNetworks/PostProcessing.py): Functions for visualizing the testing and predictions
- [Predictions.py](/NeuralNetworks/Predictions.py): Functions that perform iterative predictions for the different types of neural networks
- [PreProcessing.py](/NeuralNetworks/PreProcessing.py): All the data pre-processing functions required to construct a `LearningInstance` object for each country

### Utilities

- [DataUtils.py](/Utils/DataUtils.py): Utility functions for rea- [HierarchicalClustering.py](/Clustering/HierarchicalClustering.py): Functionality for constracting tree structures for hierarchical clustering using different linkage methods and cutting the hierarchy tree to derive the optimal number of clustersding and writing data to files.
- [PostProcessing.py](/Clustering/PostProcessing.py): Functionality for processing the results of all algorithms.
- [PreProcessing.py](/Clustering/PreProcessing.py): The pre-processing functions required to construct the final dataset for all algorithms
- [TimeSeriesUtilities.py](/Utils/TimeSeriesUtils.py): Utility functions for time series processing
- [VisualUtils.py](/Utils/VisualUtils.py): Utility functions for visualizing results