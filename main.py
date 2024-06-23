import numpy as np
import csv
from PreProcessing import preprocess_data, learning_set
from LearningInstance import LearningInstance
from NeuralNetworks import MLP, predict
from PostProcessing import plot_test_and_prediction

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
    learning_sets: list[LearningInstance] = learning_set(lags, 0.8, gdp, log_gdp, low_gdp, countries)
    inst: LearningInstance = learning_sets[countries.index("USA")] 
    model, loss, test_predictions = MLP(inst, [256,128,64,32], ["relu", "relu", "relu", "relu"])
    print(f"Loss: {loss}")
    to_predict = 50
    predictions = predict(model, inst, lags, to_predict)
    plot_test_and_prediction(inst, test_predictions, predictions)

if __name__=="__main__":
    main()