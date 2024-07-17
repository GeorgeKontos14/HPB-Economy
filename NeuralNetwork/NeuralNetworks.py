import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from NeuralNetwork.LearningInstance import LearningInstance
from NeuralNetwork.ConstructModels import construct_MLP, construct_RNN, construct_LSTM, construct_GRU, construct_BiNN

def MLP(inst: LearningInstance, 
        layers: list[int], 
        activations: list[str],
        info: int = 0):
    """
    Performs construction, training and testing of a Multi-Layer Perceptron
    with the specified configurations.

    Parameters:
    inst (LearningInstance): the learning set for the model
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use
    info (int): amount of information to be displayed by training/ testing

    Returns:
    Sequential: the fitted model
    float: the mse loss on the test set
    np.ndarray: the predictions on the test set
    """
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
    """
    Performs construction, training and testing of a Recurrent Neural Network
    with the specified configurations.

    Parameters:
    inst (LearningInstance): the learning set for the model
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use
    info (int): amount of information to be displayed by training/ testing

    Returns:
    Sequential: the fitted model
    float: the mse loss on the test set
    np.ndarray: the predictions on the test set
    """
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
    """
    Performs construction, training and testing of a Recurrent Neural Network
    using Long Short Term Memory neurons with the specified configurations.

    Parameters:
    inst (LearningInstance): the learning set for the model
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use
    info (int): amount of information to be displayed by training/ testing

    Returns:
    Sequential: the fitted model
    float: the mse loss on the test set
    np.ndarray: the predictions on the test set
    """
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
    """
    Performs construction, training and testing of a Recurrent Neural Network
    using Gated Recurrent Unit neurons with the specified configurations.

    Parameters:
    inst (LearningInstance): the learning set for the model
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use
    info (int): amount of information to be displayed by training/ testing

    Returns:
    Sequential: the fitted model
    float: the mse loss on the test set
    np.ndarray: the predictions on the test set
    """
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
    """
    Performs construction, training and testing of a Bidirectional Neural Network
    with the specified configurations.

    Parameters:
    inst (LearningInstance): the learning set for the model
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use
    layer_types (list[int]): the type of neuron each layer should use: 0 for Simple RNN;
    1 for GRU; 2 for LSTM
    info (int): amount of information to be displayed by training/ testing

    Returns:
    Sequential: the fitted model
    float: the mse loss on the test set
    np.ndarray: the predictions on the test set
    """
    model: Sequential = construct_BiNN((inst.x_train.shape[1],1), layers, activations, layer_types)
    if info > 0:
        print(model.summary())
    model.fit(inst.x_train, inst.y_train, epochs=100, batch_size=1, verbose=info, validation_data=(inst.x_test, inst.y_test))
    loss = model.evaluate(inst.x_test, inst.y_test, verbose=info)
    test_predictions = model.predict(inst.x_test, verbose=info).T[0]
    return model, loss, test_predictions