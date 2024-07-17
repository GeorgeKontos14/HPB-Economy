import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork.LearningInstance import LearningInstance

def plot_test(instance: LearningInstance, predictions: np.ndarray):
    """
    Plots the predictions on the test set

    Parameters:
    instance (LearningInstance): the learning set evaluation has been performed on
    predictions (np.ndarray): the predictions on the test set
    """
    plt.figure(figsize=(15,5))
    plt.title(f"Test for {instance.country}")
    plt.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    plt.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    plt.plot(range(2018-len(predictions), 2018), predictions, color="red", label="Test Set Prediction")
    plt.legend()
    plt.show()

def plot_test_and_prediction(instance: LearningInstance, 
                             test_predictions: np.ndarray, 
                             future_predictions: np.ndarray):
    """
    Plotsthe predictions of an ML model both for the test set and the iterative forecast

    Parameters:
    instance (LearningInstance): the learning set used
    test_predictions (np.ndarray): the predictions among the test set
    future_predictions (np.ndarray): the iterative forecast 
    """
    horizon = len(future_predictions)
    plt.figure(figsize=(15,5))
    plt.title(f"Predictions for {instance.country}")
    plt.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    plt.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    plt.plot(range(2018-len(test_predictions), 2018), test_predictions, color="red", label="Test Set Prediction")
    plt.plot(range(2018, 2018+horizon), future_predictions, color="orange", label=f"Prediction 2018-{2018+horizon}")
    plt.legend()
    plt.show()

def plot_all_tests(instance: LearningInstance,
                   test_mlp: np.ndarray,
                   test_rnn: np.ndarray,
                   test_gru: np.ndarray,
                   test_lstm: np.ndarray,
                   test_binn: np.ndarray):
    """
    Plots the test predictions for all models in different subplots

    Parameters:
    instance (LearningInstance): the learning set used
    test_mlp (np.ndarray): test predictions for MLP
    test_rnn (np.ndarray): test predictions for RNN
    test_gru (np.ndarray): test predictions for GRU
    test_lstm (np.ndarray): test predictions for LSTM
    test_binn (np.ndarray): test predictions for BiNN
    """
    fig = plt.figure(figsize=(20, 10))
    plt.title(f"Tests for {instance.country}")
    gs = fig.add_gridspec(3,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,:])

    ax1.set_title("MLP")
    ax1.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    ax1.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    ax1.plot(range(2018-len(test_mlp), 2018), test_mlp, color="red", label="MLP Predictions on the Test set")
    ax1.legend()

    ax2.set_title("RNN")
    ax2.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    ax2.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    ax2.plot(range(2018-len(test_rnn), 2018), test_rnn, color="purple", label="RNN Predictions on the Test set")
    ax2.legend()

    ax3.set_title("GRU")
    ax3.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    ax3.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    ax3.plot(range(2018-len(test_gru), 2018), test_gru, color="orange", label="GRU Predictions on the Test set")
    ax3.legend()

    ax4.set_title("LSTM")
    ax4.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    ax4.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    ax4.plot(range(2018-len(test_lstm), 2018), test_lstm, color="brown", label="LSTM Predictions on the Test set")
    ax4.legend()

    ax5.set_title("BiNN")
    ax5.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    ax5.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    ax5.plot(range(2018-len(test_binn), 2018), test_binn, color="yellow", label="BiNN Predictions on the Test set")
    ax5.legend()

    fig.tight_layout()

    plt.show()

def plot_all_predictions(instance: LearningInstance,
                         pred_mlp: np.ndarray,
                         pred_rnn: np.ndarray,
                         pred_gru: np.ndarray,
                         pred_lstm: np.ndarray,
                         pred_binn: np.ndarray):
    """
    Plots the forecasts for all models in the same plot

    Parameters:
    instance (LearningInstance): the learning set used
    pred_mlp (np.ndarray): Predictions made by MLP
    pred_rnn (np.ndarray): Predictions made by RNN
    pred_gru (np.ndarray): Predictions made by GRU
    pred_lstm (np.ndarray): Predictions made by LSTM
    pred_binn (np.ndarray): Predictions made by BiNN
    """
    to_predict = len(pred_mlp)
    plt.figure(figsize=(20,8))
    plt.title(f"Predictions for {instance.country} from 2018 to {2018+to_predict}")
    plt.plot(range(instance.start_year, 2018), instance.log_y, color="blue", label="Log GDP")
    plt.plot(range(instance.start_year, 2018), instance.low_freq, color="green", label="Low Freq. Trend of Log GDP")
    plt.plot(range(2018, 2018+to_predict), pred_mlp, color="red", label=f"MLP")
    plt.plot(range(2018, 2018+to_predict), pred_rnn, color="purple", label=f"RNN")
    plt.plot(range(2018, 2018+to_predict), pred_gru, color="orange", label=f"GRU")
    plt.plot(range(2018, 2018+to_predict), pred_lstm, color="brown", label=f"LSTM")
    plt.plot(range(2018, 2018+to_predict), pred_binn, color="yellow", label=f"BiNN")
    plt.legend()
    plt.show()