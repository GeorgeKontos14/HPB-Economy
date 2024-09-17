import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
import matplotlib.pyplot as plt
from NeuralNetworks.LearningInstance import LearningInstance
import csv

def predict_MLP(model: Sequential, 
            inst: LearningInstance, 
            lags: int, 
            horizon: int
            ) -> np.ndarray:
    """
    Performs iterative predictions for a given neural network

    Parameters:
    model (Sequential): The neural network with which predictions are performed
    inst (LearningInstance): The learning instance (country) to predict for
    lags (int): the amount of previous values to be considered
    horizon (int): the amount of future values to be predicted

    Returns:
    np.ndarray: The forecasted values
    """
    last = inst.low_freq[-lags:]
    future = np.zeros(horizon)
    for i in range(horizon):
        p = model.predict(last.reshape(1,-1), verbose=0)[0][0]
        future[i] = p
        last = np.roll(last,-1)
        last[-1] = p
    return future

def predict_RNN(model: Sequential, 
            inst: LearningInstance, 
            lags: int, 
            horizon: int
            ) -> np.ndarray:
    """
    Performs iterative predictions for a given neural network

    Parameters:
    model (Sequential): The neural network with which predictions are performed
    inst (LearningInstance): The learning instance (country) to predict for
    lags (int): the amount of previous values to be considered
    horizon (int): the amount of future values to be predicted

    Returns:
    np.ndarray: The forecasted values
    """
    last = inst.low_freq[-lags:]
    future = np.zeros(horizon)
    for i in range(horizon):
        p = model.predict(last.reshape(1,lags,1), verbose=0)[0][0]
        future[i] = p
        last = np.roll(last,-1)
        last[-1] = p
    return future
