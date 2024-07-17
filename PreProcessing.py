import numpy as np
import statsmodels.api as sm
from LearningInstance import LearningInstance

def largest_eigenvecs(A: np.ndarray, n: int):
    """
    Finds the eigenvectors of a matrix that correspond to the highest eigenvalues

    Parameters:
    A (np.ndarray): the matrix for which the eigenvectors are found
    n (int): the number of eigenvectors to be kept

    Returns:
    np.ndarray: the n largest eigenvalues of A
    np.ndarray: the n eigenvectors that correspond to the largest eigenvalues
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    indices = eigvals.argsort()[::-1][:n]
    return eigvals[indices], eigvecs[:, indices]

def baseline(T: int, 
             q: int, 
             q0: int):
    """
    Calculates baseline trigonometric regressors

    Parameters:
    T (int): Maximum length of time series
    q (int): Baseline frequency
    q0 (int): Frequency of low-frequency trends

    Returns:
    np.ndarray: The Tx2 matrix containing the constant and linear regressors
    np.ndarray: The random-walk matrix
    R: The baseline regressors
    cutoff: The cutoff frequency
    """
    X = np.zeros((T,2))
    R = np.zeros((T,q+1))
    X[:,0] = 1/T
    X[:,1] = np.array([i/T for i in range(1,T+1)])-(T+1)/2
    rows = np.arange(1,T+1).reshape(-1,1)
    cols = np.arange(1,T+1).reshape(1,-1)
    V = np.minimum(rows,cols)
    m = np.eye(T)-X@np.linalg.inv(X.T@X)@X.T
    evals, evecs = largest_eigenvecs(m@V@m, q-1)
    cutoff = 0.9999*evals[q0-2]
    R[:, :2] = X
    R[:, 2:] = evecs
    return X, V, R, cutoff

def check_nan(y_country: np.ndarray,
              T: int):
    """
    Checks for isolated not-nan values in an array

    Parameters:
    y_country (np.ndarray): The time series to be checked
    T (int): The length of the time series

    Returns:
    np.ndarray: A boolean array indicating which values are not nan or not isolated 
    """
    check = ~np.isnan(y_country)
    notnans = []
    notnans.append(check[0])
    for i in range(1,T-1):
        notnans.append(check[i-1] or check[i+1])
    notnans.append(check[T-1])
    return np.array(notnans)

def get_weights(sel: np.ndarray,
                T: int,
                X: np.ndarray,
                V: np.ndarray,
                cutoff: float,
                q0: int):
    """
    Gets regressor weights for a specified country

    Parameters:
    sel (np.ndarray): The array indicating which values are not nan
    T (int): The length of the time series
    X (np.ndarray): The non-trigonometric trends
    V (np.ndarray): The random-walk matrix
    cutoff (float): The cutoff frequency
    q0 (int): The frequency of the low frequency trends

    Returns:
    np.ndarray: The calculated regressor weigths
    """
    X_n = np.zeros_like(X)
    for i in range(2):
        X_n[:,i] = X[:,i]*sel
    m = np.diag(sel)-X_n@np.linalg.inv(X_n.T@X_n)@X_n.T
    evals, evecs = largest_eigenvecs(m@V@m, q0-1)
    qw = np.sum(evals>cutoff)
    w = np.zeros((T,qw+2))
    w[:, :2] = X_n
    w[:, 2:] = evecs[:, :qw]
    return w

def find_country_trend(y_country: np.ndarray,
                       T: int,
                       X: np.ndarray,
                       V: np.ndarray,
                       R: np.ndarray, 
                       cutoff: float,
                       q:int,
                       q0: int):
    """
    Finds the low frequency trend for a specified country

    Parameters:
    y_country (np.ndarray): The annual GDP per capita for the country
    T (int): The length of the time series
    X (np.ndarray): The constant and linear trends
    V (np.ndarray): The random-walk matrix
    R (np.ndarray): The baseline regressors
    cutoff (float): The cutoff frequency
    q (int): Baseline frequency
    q0 (int): Frequency of low-frequency trends

    Returns:
    np.ndarray: The annual log GDP per capita
    np.ndarray: The projection matrix to the baseline regressors
    np.ndarray: The low-frequency regressors of the country
    """
    notnans = check_nan(y_country, T)
    w = get_weights(notnans, T, X, V, cutoff, q0)
    AB = np.zeros((q+1,q+1))
    for i in range(w.shape[1]):
        model = sm.OLS(w[:,i], R)
        results = model.fit()
        AB[i,:] = results.params
    AB[(w.shape[1]+1):, :] = 0
    for i in range(w.shape[1],q+1):
        AB[i][i] = 0
    AB = AB.T
    A = AB[:,:w.shape[1]]
    AApAi = A@np.linalg.inv(A.T@A)
    filtered = np.where(np.isnan(y_country), 1, y_country)
    log_y = np.log(filtered)
    Y = w.T@log_y
    return log_y, AApAi, Y

def preprocess_data(gdp: np.ndarray,
                    T: int,
                    q: int,
                    q0: int):
    """
    Performs the necessary data pre-processing for all countries

    Parameters:
    gdp (np.ndarray): A (n, T) array containing the annual GDP for each country
    T (int): The years for which the data is gathered
    q (int): Baseline frequency
    q0 (int): Frequency of low-frequency trends

    Returns:
    np.ndarray: The annual log GDP for each country
    np.ndarray: The low-frequency trends of the annual log GDP for each country
    """
    X,V,R,cutoff = baseline(T,q,q0)
    R_inv = np.linalg.pinv(R.T)
    log_gdp = np.zeros_like(gdp)
    low_gdp = np.zeros_like(gdp)
    for i, y in enumerate(gdp):
        log_y, AApAi, Y = find_country_trend(y, T, X, V, R, cutoff, q, q0)
        log_gdp[i] = log_y
        low_gdp[i] = R_inv@AApAi@Y
    return log_gdp, low_gdp

def learning_set(lags: int, 
                 split: float, 
                 gdp: np.ndarray,
                 log_gdp: np.ndarray,
                 low_gdp: np.ndarray, 
                 countries: list[str]
                ) -> list[LearningInstance]:
    """
    Creates the learning set for each country

    Parameters:
    lags (int): The amount of previous values to be considered as features
    split (float): The training-testing split to be used
    gdp (np.ndarray): A (n, T) array containing the annual GDP for each country
    log_gdp (np.ndarray): The annual log GDP for each country
    low_gdp (np.ndarray): The low-frequency trends of the annual log GDP for each country
    countries (list[str]): The country names

    Returns:
    list[LearningInstance]: A list containing the learning set for each country
    """
    result: list[LearningInstance] = []
    for i, data in enumerate(low_gdp):
        first_ind = np.where(~np.isnan(gdp[i]))[0][0]
        filtered = data[first_ind:]
        num_datapoints = len(filtered)-lags
        x = np.array([filtered[j:j+lags] for j in range(num_datapoints)])
        y = np.array([filtered[j+lags] for j in range(num_datapoints)])
        train_size = int(len(x)*split)
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        result.append(LearningInstance(
            countries[i],
            x_train,
            y_train,
            x_test,
            y_test,
            log_gdp[i][first_ind:],
            filtered
        ))
    
    return result
