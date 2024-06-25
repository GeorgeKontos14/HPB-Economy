import numpy as np
import csv
from PreProcessing import preprocess_data, learning_set
from LearningInstance import LearningInstance
from NeuralNetworks import MLP, RNN, BiNN, GRU, LSTM, predict
from PostProcessing import plot_test_and_prediction, plot_all_tests, plot_all_predictions

n = 113
T = 118
q = 31
q0 = 16
names_path = "Data/names.txt"
pop_path = "Data/pop_raw.csv"
gdp_path = "Data/yp_raw.csv"

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
    inst: LearningInstance = learning_sets[countries.index("USA")] 
    layers = [128,64,32]
    activations = ['relu', 'relu', 'relu']
    layer_types = [2, 2, 2]
    mlp, loss_mlp, test_mlp = MLP(inst, layers, activations, info=1)
    rnn, loss_rnn, test_rnn = RNN(inst, layers, activations, info=1)
    gru, loss_gru, test_gru = GRU(inst, layers, activations, info=1)
    lstm, loss_lstm, test_lstm = LSTM(inst, layers, activations, info=1)
    binn, loss_binn, test_binn = BiNN(inst, layers, activations, layer_types, info=1)
    print(f"MLP Loss: {loss_mlp}")
    print(f"RNN Loss: {loss_rnn}")
    print(f"GRU Loss: {loss_gru}")
    print(f"LSTM Loss: {loss_lstm}")
    print(f"BiNN Loss: {loss_binn}")
    plot_all_tests(inst, test_mlp, test_rnn, test_gru, test_lstm, test_binn)
    horizon = 50
    pred_mlp = predict(mlp, inst, lags, horizon)
    pred_rnn = predict(rnn, inst, lags, horizon)
    pred_gru = predict(gru, inst, lags, horizon)
    pred_lstm = predict(lstm, inst, lags, horizon)
    pred_binn = predict(binn, inst, lags, horizon)
    plot_all_predictions(inst, pred_mlp, pred_rnn, pred_gru, pred_lstm, pred_binn)

if __name__=="__main__":
    main()