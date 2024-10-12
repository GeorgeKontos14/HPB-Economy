import numpy as np

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight

from tslearn.clustering import KernelKMeans
from tslearn.clustering.utils import _check_no_empty_cluster, EmptyClusterError
from tslearn.metrics import sigma_gak
from tslearn.utils import check_dims

class NonRandomKernelKMeans(KernelKMeans):
    """Performs Kernel k-means clustering with nonrandom initialization

    Parameters:
        n_clusters (int): Number of clusters to form
        init_labels (np.ndarray): Labels to start the algorithm with
    """

    def __init__(self, n_clusters: int, init_labels: np.ndarray):
        KernelKMeans.__init__(
            self, n_clusters=n_clusters, kernel_params={"sigma": "auto"}, verbose=0
        )
        self.init_labels = init_labels
    
    def _fit_one_init(self, K, rs):
        n_samples = K.shape[0]

        self.labels_ = self.init_labels

        dist = np.empty((n_samples, self.n_clusters))
        old_inertia = np.inf

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist)
            self.labels_ = dist.argmin(axis=1)
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            self.inertia_ = self._compute_inertia(dist)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if np.abs(old_inertia - self.inertia_) < self.tol:
                break
            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def fit(self, X, y=None, sample_weight=None):
        """Compute kernel k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored
        sample_weight : array-like of shape=(n_ts, ) or None (default: None)
            Weights to be given to time series in the learning process. By
            default, all time series weights are equal.
        """

        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)

        sample_weight = _check_sample_weight(sample_weight=sample_weight, X=X)

        max_attempts = max(self.n_init, 10)
        kernel_params = self._get_kernel_params()
        if self.kernel == "gak":
            self.sigma_gak_ = kernel_params.get("sigma", 1.0)
            if self.sigma_gak_ == "auto":
                self.sigma_gak_ = sigma_gak(X)
        else:
            self.sigma_gak_ = None

        self.labels_ = self.init_labels
        self.inertia_ = None
        self.sample_weight_ = None
        self._X_fit = None
        # n_iter_ will contain the number of iterations the most
        # successful run required.
        self.n_iter_ = 0

        n_samples = X.shape[0]
        K = self._get_kernel(X)
        sw = sample_weight if sample_weight is not None else np.ones(n_samples)
        self.sample_weight_ = sw
        rs = check_random_state(self.random_state)

        last_correct_labels = self.init_labels
        min_inertia = np.inf
        n_attempts = 0
        n_successful = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(K, rs)
                if self.inertia_ < min_inertia:
                    last_correct_labels = self.labels_
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        if n_successful > 0:
            self.labels_ = last_correct_labels
            self.inertia_ = min_inertia
            self._X_fit = X
        return self