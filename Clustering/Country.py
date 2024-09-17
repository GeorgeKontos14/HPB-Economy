import numpy as np
import statsmodels.api as sm
import scipy.stats

class Country:

    def __init__(self,
                 name: str,
                 gdp: np.ndarray,
                 longtitude: float,
                 latitude: float,
                 area: float,
                 population: np.ndarray,
                 working_population: np.ndarray,
                 currency: np.ndarray):
        """
        Object that holds all information for a country

        Parameters:
        name (str): The name of the country
        gdp (np.ndarray): The annual GDP per capita of the country
        longtitude (float): The geographical longtitude of the country
        latitude (float): The geographical latitude of the country
        area (float): The land area of the country
        population (np.ndarray): The annual population of the country
        working_population (np.ndarray): The annual labor partitipation percentage
        currency (np.ndarray): The annual exchange rate of the currency of the country
        """
        self.name = name
        self.gdp = gdp
        self.longitude = longtitude
        self.latitude = latitude
        self.area = area
        self.population = population
        self.working_population = working_population
        self.currency = currency
        self.cluster = -1 # The cluster the country is included in after the clustering

    def feature_vector(self, h: int = 1) -> np.ndarray:
        """
        Calculates the feature vector for the country. For each time series involved
        in the calculations, 6 features are computed: mean, variance, lag autocovariance,
        lag autocorrelation, kurtosis and skewness

        Parameters:
        h (int): The number of lags to be considered for autocovariance and autocorrelation

        Returns:
        np.ndarray: The feature vector for the specified country
        """
        features = np.zeros(12)

        # Features regarding the gdp:
        features[:6] = time_series_features(self.gdp, h)

        # Features regarding the geographical locations:
        features[6] = self.longitude
        features[7] = self.latitude

        # Features regarding the area:
        features[8] = self.area[-1]
        features[9] = np.nanmax(self.area)-np.nanmin(self.area)

        # Features regarding population and labor force:
        features[10] = np.mean(self.population)
        features[11] = np.var(self.population)



        return features
    
def time_series_features(y: np.ndarray, h: int = 1) -> np.ndarray:
    """
    Calculates the features for a time series, including the mean, variance, autocorrelation,
    autocovariance, kurtosis and skewness

    Parameters:
    y (np.ndarray): The time series in question
    h (int): The number of lags to be used for autocorrelation and autocovariance

    Returns:
    np.ndarray: The feature vector for the time series
    """
    feature_vector = np.zeros(6)
    m = len(y)
    mean = np.mean(y) 

    feature_vector[0] = mean
    feature_vector[1] = np.var(y) # variance
    feature_vector[2] = np.sum((y[:m-h]-mean)*(y[h:]-mean))/m # lag-h autocovariance
    feature_vector[3] = sm.tsa.acf(y)[h] # lag-h autocorrelation
    feature_vector[4] = scipy.stats.kurtosis(y) # kurtosis
    feature_vector[5] = scipy.stats.skew(y) # skewness

    return feature_vector