import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from LearningInstance import LearningInstance
from ConstructModels import construct_MLP, construct_RNN, construct_LSTM, construct_GRU, construct_BiNN

def MLP(inst: LearningInstance, 
        layers: list[int], 
        activations: list[str],
        info: int = 0):
    model: Sequential = construct_MLP(inst.x_train.shape[1:], layers, activations)
    if info > 0:
        print(model.summary())
    model.fit(inst.x_train, inst.y_train, epochs=100, batch_size=1, verbose=info, validation_data=(inst.x_test, inst.y_test))
    loss = model.evaluate(inst.x_test, inst.y_test, verbose=info)
    test_predictions = model.predict(inst.x_test, verbose=info).T[0]
    return model, loss, test_predictions

def RNN(inst: LearningInstance,
        layers: list[int],
        activations: list[str],
        info: int = 0):
    model: Sequential = construct_RNN((inst.x_train.shape[1],1), layers, activations)
    if info > 0:
        print(model.summary())
    model.fit(inst.x_train, inst.y_train, epochs=100, batch_size=1, verbose=info, validation_data=(inst.x_test, inst.y_test))
    loss = model.evaluate(inst.x_test, inst.y_test, verbose=info)
    test_predictions = model.predict(inst.x_test, verbose=info).T[0]
    return model, loss, test_predictions

def LSTM(inst: LearningInstance,
        layers: list[int],
        activations: list[str],
        info: int = 0):
    model: Sequential = construct_LSTM((inst.x_train.shape[1],1), layers, activations)
    if info > 0:
        print(model.summary())
    model.fit(inst.x_train, inst.y_train, epochs=100, batch_size=1, verbose=info, validation_data=(inst.x_test, inst.y_test))
    loss = model.evaluate(inst.x_test, inst.y_test, verbose=info)
    test_predictions = model.predict(inst.x_test, verbose=info).T[0]
    return model, loss, test_predictions

def GRU(inst: LearningInstance,
        layers: list[int],
        activations: list[str],
        info: int = 0):
    model: Sequential = construct_GRU((inst.x_train.shape[1],1), layers, activations)
    if info > 0:
        print(model.summary())
    model.fit(inst.x_train, inst.y_train, epochs=100, batch_size=1, verbose=info, validation_data=(inst.x_test, inst.y_test))
    loss = model.evaluate(inst.x_test, inst.y_test, verbose=info)
    test_predictions = model.predict(inst.x_test, verbose=info).T[0]
    return model, loss, test_predictions

def BiNN(inst: LearningInstance,
        layers: list[int],
        activations: list[str],
        layer_types: list[int],
        info: int = 0):
    model: Sequential = construct_BiNN((inst.x_train.shape[1],1), layers, activations, layer_types)
    if info > 0:
        print(model.summary())
    model.fit(inst.x_train, inst.y_train, epochs=100, batch_size=1, verbose=info, validation_data=(inst.x_test, inst.y_test))
    loss = model.evaluate(inst.x_test, inst.y_test, verbose=info)
    test_predictions = model.predict(inst.x_test, verbose=info).T[0]
    return model, loss, test_predictions

def predict(model: Sequential, 
            inst: LearningInstance, 
            lags: int, 
            horizon: int
            ) -> np.ndarray:
    last = inst.low_freq[-lags:]
    future = np.zeros(horizon)
    for i in range(horizon):
        p = model.predict(last.reshape(1,-1), verbose=0)[0][0]
        future[i] = p
        last = np.roll(last,-1)
        last[-1] = p
    return future