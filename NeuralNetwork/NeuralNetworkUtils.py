import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
import matplotlib.pyplot as plt
from NeuralNetwork.LearningInstance import LearningInstance
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

# def run_all_countries(model: Sequential,
#                       learning_sets: list[LearningInstance],
#                       info: int = 0,
#                       out: str = None,
#                       plot_path: str = None,
#                       model_name: str = None,
#                       model_desc: str = None):
#     """
#     Trains and Tests a model for all countries and saves the results

#     Parameters:
#     model (Sequential): the model to fit
#     learning_sets (list[LearningInstance]): the learning sets for all the countries
#     info (int): the amount of information to be displayed for fitting/evaluating
#     out (str): the path for the file to write results to
#     plot_path (str): the path where the loss plot for all countries should be saved
#     model_name (str): the name of the model (for saving purposes)
#     model_desc (str): the model description (for saving purposes)

#     Returns:
#     list[float]: the loss for each country
#     float: the average loss across all countries
#     """
#     loss_list = []
#     avg_loss = 0
#     for inst in learning_sets:
#         if info > 0:
#             print(inst.country)
#         model.fit(inst.x_train, inst.y_train, epochs=100, verbose=0, validation_data=(inst.x_test, inst.y_test))
#         loss = model.evaluate(inst.x_test, inst.y_test, verbose=0)
#         loss_list.append(loss)
#         avg_loss += loss
#     avg_loss = avg_loss/len(learning_sets)
#     if out is not None:
#         with open(out, 'a') as file:
#             file.write('\n\n')
#             file.write(f"{model_name}\n")
#             file.write(f"{model_desc}\n")
#             file.write(f"Average Loss: {avg_loss}")
#         plt.title("Loss per country")
#         plt.figure(figsize=(20,5))
#         plt.plot(loss_list, "o-")
#         plt.savefig(f"{plot_path+model_name}.png")
#     return loss_list, avg_loss

# def fit_and_predict_all_countries(model: Sequential,
#                                   learning_sets: list[LearningInstance],
#                                   lags: int,
#                                   horizon: int,
#                                   info: int=0,
#                                   out: str = None):
#     """
#     Trains a model and performs predictions for all countries

#     Parameters:
#     model (Sequential): the model to fit
#     learning_sets (list[LearningInstance]): the learning sets for all the countries
#     lags (int): the amount of previous values to be considered
#     horizon (int): the amount of future values to be predicted
#     info (int): the amount of information to be displayed for fitting/evaluating
#     out (str): the path for the file to write results to

#     Returns:
#     np.ndarray: An (n, horizon) 2D array where each row contains the prediction for each country
#     """
#     n = len(learning_sets)
#     predictions = np.zeros((n,horizon))
#     for i, inst in enumerate(learning_sets):
#         if info > 0:
#             print(inst.country)
#         model.fit(inst.x_train, inst.y_train, epochs=100, verbose=0, validation_data=(inst.x_test, inst.y_test))
#         predictions[i] = predict(model, inst, lags, horizon)
#     if out is not None:
#         with open(out, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             for row in predictions:
#                 vals = [float(val) for val in row]    
#                 writer.writerow(vals)
#             file.write('\n')
#     return predictions