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

### [Forecasting](/prob_forecasting.ipynb)

In order to develop an understanding of the future values of the GDP growth of each country, probabilistic forecasts are utilized. Probabilistic forecasting is prefered over point forecasting, as it provides intel about the ranges within the future values are most likely to fall and the model's confidence. This approach works better than point forecasting when a very limited amount of data is available, which is the case in this situation.

Throughout the experiments, Gradient Boosting Regressors are used to optimize the predictions. Univariate and independent multi-series predictions are performed recursively, while multivariate predictions are performed directly, both using `skforecast`'s framework.

For these forecasts, three different hyperparameters shall be predetermined before training and testing the forecast:
- `lags`: The number of previous values to be considered for the prediction of the next value.
- `window_size`: The size of the window over which additional statistics are calculated.
- `differentiation`: The number of times the time series shall be differenced

One can easily observe that these hyperparameters resemble the order tuple for an $ARIMA(p,d,q)$ process. For this reason, the parameters are determined by the optimal corresponding $(p,d,q)$ orders. In general, multivariate models allow different numbers of lags for each variable, but demand a unique order of differentiation and stats window size. The mode of the differentiation orders and moving average orders of the groups of countries considered is used.

![Overview of ARIMA orders](/Images/Forecasts/arima_orders.png)

#### Univariate Forecasting

In the univariate forecasts, each time series is considered in isolation and a gradient boosting regressor is fitted to perform probabilistic predictions on the series in question. The figure below showcases the performance of such a model for the GDP evolution of Serbia, when the modeled is configured to calculate the $67\%$ prediction intervals.

![Univariate forecast for Serbia (SRB)](/Images/Forecasts/univariate_srb.png)

#### Independent Multi-Series Forecasting

It is possible to perform imultaneous probabilistic recursive predictions for multiple time series, without considering cross-series dependencies. Modelling multiple time series together adds to the robustness of the model, epsecially under the lack of data for the specific experiment. First, we consider the entire dataset at once. Afterwards, we consider the clusters produced by the use of Kernel $13$-Means on the non-zero-mean unit-variance dataset in isolation. Once again, $67\%$ prediction intervals for Serbia are shown.

![Independend multi-series forecast for Serbia (SRB) considering the entire dataset](/Images/Forecasts/independent_all_srb.png)
![Independend multi-series forecast for Serbia (SRB) considering only its cluster (i.e. Iraq, Russia, Serbia, Venezuela)](/Images/Forecasts/independent_all_srb.png)

#### Many-to-one Forecasting

The framework developed from `skforecast` has functionality for performing direct predictions where the interdependence between different variables is considered. However, the framework is limited to perform predictions on only one variable, due to efficiency limitations. We utilize this functionality by creating a separate model for each country of interest. While this approach is not fully indicative of the joint GDP growth, it provides insight on how its individual GDP progression is affected by other countries.

![Many-to-one forecast for Serbia (SRB) considering the entire dataset](/Images/Forecasts/manytoone_all_srb.png)
![Many-to-one forecast for Serbia (SRB) considering only its cluster (i.e. Iraq, Russia, Serbia, Venezuela)](/Images/Forecasts/manytoone_cluster_srb.png)

#### Many-to-many Forecasting

The `ForecastDirectMultiOutput` class adjusts the `skforecast` implementation of `ForecasterDirectMultiVariate` to make predictions on multiple levels. While this is considerably less efficient than the previous approach, it allows us to better capture cross-country dependencies. The `sklearn.multioutput.MultiOutputRegressor` wrapper is used to allow the use of Gradient Boosting Regression for multivariate predictions.

![Many-to-many forecast for Serbia (SRB) considering the entire dataset](/Images/Forecasts/manytomany_all_srb.png)
![Many-to-many forecast for Serbia (SRB) considering only its cluster (i.e. Iraq, Russia, Serbia, Venezuela)](/Images/Forecasts/manytomany_cluster_srb.png)

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
- [ForecasterMultioutput.py](/Forecasting/ForecasterMultioutput.py): Class that adjusts `skforecast.direct.ForecasterDirectMultivariate` to perform predictions on multiple time series at once
- [MultivariarteForecasts.py](/Forecasting/MultivariateForecasts.py): Functions for multivariate (recursive & direct) probabilistic forecasts.
- [UnivariateForecasts.py](/Forecasting/UnivariateForecasts.py): Functions for univariate recursive probabilistic forecasts.

### Utilities

- [DataUtils.py](/Utils/DataUtils.py): Utility functions for rea- [HierarchicalClustering.py](/Clustering/HierarchicalClustering.py): Functionality for constracting tree structures for hierarchical clustering using different linkage methods and cutting the hierarchy tree to derive the optimal number of clustersding and writing data to files.
- [PostProcessing.py](/Clustering/PostProcessing.py): Functionality for processing the results of all algorithms.
- [PreProcessing.py](/Clustering/PreProcessing.py): The pre-processing functions required to construct the final dataset for all algorithms
- [TimeSeriesUtilities.py](/Utils/TimeSeriesUtils.py): Utility functions for time series processing
- [VisualUtils.py](/Utils/VisualUtils.py): Utility functions for visualizing results