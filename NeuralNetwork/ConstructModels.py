from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, SimpleRNN # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def build_MLP(
        learning_rate: float, 
        num_layers: int, 
        num_units: int, 
        lags: int
    ) -> Sequential:
    """
    Constructs a Multi-Layer Perceptron with the specified configurations:

    Parameters:
    learning_rate (float): The learning rate of the neurons
    num_layers (int): The number of layers of the model
    num_units (int): The number of neurons in each layer
    lags (int): The number of past values to be considered for each prediction

    Returns:
    Sequential: The constructed MLP
    """
    model = Sequential()
    model.add(Input(shape=(lags,)))

    for _ in range(num_layers):
        model.add(Dense(int(num_units), activation='relu'))
    
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss = 'mean_squared_error'
    )

    return model


def build_RNN(learning_rate, num_layers, num_units, lags):
    """
    Constructs a Recurrent Neural Network with the specified configurations:

    Parameters:
    learning_rate (float): The learning rate of the neurons
    num_layers (int): The number of layers of the model
    num_units (int): The number of neurons in each layer
    lags (int): The number of past values to be considered for each prediction

    Returns:
    Sequential: The constructed MLP
    """
    model = Sequential()
    
    model.add(Input(shape=(lags,1)))

    # Add LSTM layers
    for i in range(num_layers):
            model.add(SimpleRNN(int(num_units), return_sequences=i<num_layers-1))
    
    # Output layer for regression
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    return model