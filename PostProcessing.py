import matplotlib.pyplot as plt
import numpy as np
from LearningInstance import LearningInstance

def plot_test(instance: LearningInstance, predictions: np.ndarray):
    plt.figure(figsize=(15,5))
    plt.title(f"Test for {instance.country}")
    plt.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    plt.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    plt.plot(range(2018-len(predictions), 2018), predictions, color="red", label="Prediction")
    plt.legend()
    plt.show()

def plot_test_and_prediction(instance: LearningInstance, test_predictions: np.ndarray, future_predictions: np.ndarray):
    to_predict = len(future_predictions)
    plt.figure(figsize=(15,5))
    plt.title(f"Predictions for {instance.country}")
    plt.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    plt.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    plt.plot(range(2018-len(test_predictions), 2018), test_predictions, color="red", label="Prediction")
    plt.plot(range(2018, 2018+to_predict), future_predictions, color="orange", label=f"Prediction 2018-{2018+to_predict}")
    plt.legend()
    plt.show()