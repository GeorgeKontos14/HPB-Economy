import numpy as np

class LearningInstance:
    def __init__(self, 
                 country: str, 
                 x_train: np.ndarray, 
                 y_train: np.ndarray, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray,
                 log_y: np.ndarray,
                 low_freq: np.ndarray):
        """
        Object containing all relevant information for a neural network's learning set

        Parameters:
        country (str): The country the learning set corresponds to
        x_train (np.ndarray): The training data
        y_train (np.ndarray): The ground truth for the training data
        x_test (np.ndarray): The test set
        y_test (np.ndarray): The ground truth for the test set
        log_y (np.ndarray): The log annual GDP per capita for the specified country
        low_freq (np.ndarray): The low frequency trend for the country over the course 
        of the available GDP data
        """
        self.country = country
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.log_y = log_y
        self.low_freq = low_freq
        self.start_year = 2018-len(log_y)

    def __str__(self):
        return f"Learning Set for {self.country}"