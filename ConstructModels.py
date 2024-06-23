from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.losses import mse # type: ignore
from LearningInstance import LearningInstance

def construct_MLP(input_shape: tuple, 
                  layers: list[int], 
                  activations: list[str]
                  ) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i, neurons in enumerate(layers):
        model.add(Dense(neurons, activation=activations[i]))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=mse)
    return model