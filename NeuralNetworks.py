import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from LearningInstance import LearningInstance
from ConstructModels import construct_MLP

def MLP(inst: LearningInstance, 
        layers: list[int], 
        activations: list[str],
        info: int = 0):
    model: Sequential = construct_MLP(inst.x_train.shape[1:], layers, activations)
    model.fit(inst.x_train, inst.y_train, epochs=100, batch_size=1, verbose=info)
    loss = model.evaluate(inst.x_test, inst.y_test, verbose=info)
    test_predictions = model.predict(inst.x_test, verbose=info).T[0]
    return model, loss, test_predictions

def predict(model: Sequential, 
            inst: LearningInstance, 
            lags: int, 
            to_predict: int
            ) -> np.ndarray:
    last = inst.low_freq[-lags:]
    future = np.zeros(to_predict)
    for i in range(to_predict):
        p = model.predict(last.reshape(1,-1), verbose=0)[0][0]
        future[i] = p
        last = np.roll(last,-1)
        last[-1] = p
    return future