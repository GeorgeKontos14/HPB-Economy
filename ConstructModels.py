from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU, Bidirectional # type: ignore
from tensorflow.keras.losses import mse # type: ignore
from LearningInstance import LearningInstance

def construct_MLP(input_shape: tuple, 
                  layers: list[int], 
                  activations: list[str]
                  ) -> Sequential:
    """
    Constructs a Multi-Layer Perceptron with the specified configurations

    Parameters:
    input_shape (tuple): the shape of the input
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use

    Returns:
    Sequential: the compiled model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i, neurons in enumerate(layers):
        model.add(Dense(neurons, activation=activations[i]))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=mse)
    return model

def construct_RNN(input_shape: tuple,
                  layers: list[int],
                  activations: list[str]
                  ) -> Sequential:
    """
    Constructs a Recurrent Neural Network with the specified configurations

    Parameters:
    input_shape (tuple): the shape of the input
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use

    Returns:
    Sequential: the compiled model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i, neurons in enumerate(layers):
        model.add(SimpleRNN(neurons, return_sequences=i<len(layers)-1, activation=activations[i]))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=mse)
    return model

def construct_LSTM(input_shape: tuple,
                  layers: list[int],
                  activations: list[str]
                  ) -> Sequential:
    """
    Constructs a Recurrent Neural Network using Long Short Term Memory neurons with the specified configurations

    Parameters:
    input_shape (tuple): the shape of the input
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use

    Returns:
    Sequential: the compiled model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i, neurons in enumerate(layers):
        model.add(LSTM(neurons, return_sequences=i<len(layers)-1, activation=activations[i]))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=mse)
    return model

def construct_GRU(input_shape: tuple,
                  layers: list[int],
                  activations: list[str]
                  ) -> Sequential:
    """
    Constructs a Recurrent Neural Network using Gated Recurrent Unit neurons with the specified configurations

    Parameters:
    input_shape (tuple): the shape of the input
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use

    Returns:
    Sequential: the compiled model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i, neurons in enumerate(layers):
        model.add(GRU(neurons, return_sequences=i<len(layers)-1, activation=activations[i]))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=mse)
    return model

def construct_BiNN(input_shape: tuple,
                  layers: list[int],
                  activations: list[str],
                  layer_types: list[int]
                  ) -> Sequential:
    """
    Constructs a Bidirectional Neural Network with the specified configurations

    Parameters:
    input_shape (tuple): the shape of the input
    layers (list[int]): the list containing how many neurons each layer should have
    activations (list[str]): the list with the activation function each layer should use
    layer_types (list[int]): the type of neuron each layer should use: 0 for Simple RNN;
    1 for GRU; 2 for LSTM
    
    Returns:
    Sequential: the compiled model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i, neurons in enumerate(layers):
        ret = i<len(layers)-1
        if layer_types[i] == 0:
            layer = SimpleRNN(neurons, return_sequences=ret, activation=activations[i])
        elif layer_types[i] == 1:
            layer = GRU(neurons, return_sequences=ret, activation=activations[i])
        else:
            layer = LSTM(neurons, return_sequences=ret, activation=activations[i])
        model.add(Bidirectional(layer))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=mse)
    return model