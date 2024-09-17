import numpy as np
from skopt import gp_minimize
from skopt import Real, Integer
from tensorflow.keras.models import Sequential # type: ignore

from NeuralNetworks.LearningInstance import LearningInstance
from NeuralNetworks.ConstructModels import build_MLP, build_RNN

def tune_MLP(
        instance: LearningInstance, 
        lags: int, 
        verbose: bool = False
    ) -> Sequential:
    """
    Constructs a fine-tuned Multi-Layer Perceptron based on the country data

    Parameters:
    instance (LearningInstance): the learning instance of the country in question
    lags (int): The number of past values to be considered for each prediction
    verbose (bool): Indicate whether information about the tuned model should be printed

    Returns:
    Sequential: The fine-tumed MLP
    """
    def objective(params):
        """
        The function used for the Bayesian Optimization algorithm
        """
        learning_rate, num_layers, num_units = params
        num_units = int(num_units)
        model = build_MLP(learning_rate, num_layers, num_units, lags)
        history = model.fit(
            instance.x_train, 
            instance.y_train, 
            validation_data = (instance.x_test, instance.y_test), 
            epochs = 50, 
            batch_size = 32,
            verbose = 0
        )
        val_loss = history.history['val_loss'][-1]
        # Return the validation loss (or any metric to minimize)
        return val_loss
    
    space = [Real(1e-6, 1e-2, "log-uniform", name='learning_rate'),
         Integer(1, 5, name='num_layers'),
         Integer(10, 500, name='num_units')]
    
    result = gp_minimize(objective, space, n_calls=50, random_state=0)

    learning_rate = result.x[0]
    num_layers = result.x[1]
    num_units = result.x[2]

    if verbose:
        print(f"Learning rate: {learning_rate}")
        print(f"Number of layers: {num_layers}")
        print(f"Neurons per layer: {num_units}")
    
    return build_MLP(learning_rate, num_layers, num_units, lags)

def tune_RNN(
        instance: LearningInstance, 
        lags: int, 
        verbose: bool = False
    ) -> Sequential:
    """
    Constructs a fine-tuned Recurrent Neural Network
    Parameters:
    instance (LearningInstance): the learning instance of the country in question
    lags (int): The number of past values to be considered for each prediction
    verbose (bool): Indicate whether information about the tuned model should be printed

    Returns:
    Sequential: The fine-tuned RNN
    """
    def objective(params):
        """
        The function used for the Bayesian Optimization algorithm
        """
        learning_rate, num_layers, num_units = params
        num_units = int(num_units)
        model = build_RNN(learning_rate, num_layers, num_units, lags)
        history = model.fit(
            instance.x_train.reshape(instance.x_train.shape[0], instance.x_train.shape[1], 1), 
            instance.y_train, 
            validation_data = (instance.x_test.reshape(instance.x_test.shape[0], instance.x_test.shape[1],1), instance.y_test), 
            epochs = 50, 
            batch_size = 32,
            verbose = 0
        )
        val_loss = history.history['val_loss'][-1]
        # Return the validation loss (or any metric to minimize)
        return val_loss
    
    space = [Real(1e-6, 1e-2, "log-uniform", name='learning_rate'),
         Integer(1, 5, name='num_layers'),
         Integer(10, 500, name='num_units')]
    
    result = gp_minimize(objective, space, n_calls=50, random_state=0)

    learning_rate = result.x[0]
    num_layers = result.x[1]
    num_units = result.x[2]

    if verbose:
        print(f"Learning rate: {learning_rate}")
        print(f"Number of layers: {num_layers}")
        print(f"Neurons per layer: {num_units}")
    
    return build_RNN(learning_rate, num_layers, num_units, lags)