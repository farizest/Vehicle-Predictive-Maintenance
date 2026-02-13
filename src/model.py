
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Input, Layer, LSTM, Dense, Dropout
import tensorflow.keras.backend as K

class Attention(Layer):
    """Custom Attention Layer."""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        return K.sum(context, axis=1)
    
    def get_config(self):
        return super(Attention, self).get_config()

def build_model(window_size, n_features):
    """Builds the BiLSTM + Attention model."""
    inputs = Input(shape=(window_size, n_features))
    
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    x = Attention()(x)
    
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    
    return model
