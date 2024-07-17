import numpy as np
import csv
from PreProcessing import preprocess_data, learning_set
from LearningInstance import LearningInstance
from ConstructModels import construct_BiNN, construct_LSTM, construct_MLP, construct_RNN, construct_GRU
from NeuralNetworks import MLP, RNN, BiNN, GRU, LSTM
from Utils.NeuralNetworkUtils import predict, run_all_countries, fit_and_predict_all_countries
from PostProcessing import plot_test_and_prediction, plot_all_tests, plot_all_predictions

n = 113
T = 118
q = 31
q0 = 16
names_path = "Data/names.txt"
pop_path = "Data/pop_raw.csv"
gdp_path = "Data/yp_raw.csv"
nn_predictions = "nn_predictions.csv"

def main():
    countries = []
    with open(names_path, 'r') as file:
        rows = file.readlines()
        for row in rows:
            countries.append(row[:3])
    gdp = np.zeros((n,T))
    with open(gdp_path, 'r') as file:
        rows = csv.reader(file)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                gdp[j][i] = float(val)
    
    lags = 10
    log_gdp, low_gdp = preprocess_data(gdp, T, q, q0)
    learning_sets: list[LearningInstance] = learning_set(lags, 0.7, gdp, log_gdp, low_gdp, countries)

    config = [([64], ["relu"]), ([64, 64], ["relu", "relu"]), ([96, 64, 32], ["relu", "relu", "relu"])]
    names = ["MLP", "RNN", "GRU", "LSTM", "BiNN with SimpleRNN", "BiNN with GRU", "BiNN with LSTM"]
    for layers, activations in config:
        print(layers, activations)
        mlp = construct_MLP(learning_sets[0].x_train.shape[1:], layers, activations)
        rnn = construct_RNN((lags, 1), layers, activations)
        gru = construct_GRU((lags, 1), layers, activations)
        lstm = construct_LSTM((lags, 1), layers, activations)
        binnR = construct_BiNN((lags, 1), layers, activations, [0 for _ in range(len(layers))])
        binnG = construct_BiNN((lags, 1), layers, activations, [1 for _ in range(len(layers))])
        binnL = construct_BiNN((lags, 1), layers, activations, [2 for _ in range(len(layers))])
        models = [mlp, rnn, gru, lstm, binnR, binnG, binnL]
        for i,model in enumerate(models):
            print(f"--------------------{names[i]}--------------------")
            fit_and_predict_all_countries(model, learning_sets, lags, 100, 1, nn_predictions)


if __name__=="__main__":
    main()