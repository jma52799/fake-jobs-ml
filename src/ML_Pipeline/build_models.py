from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from ML_Pipeline.constants import *

def build_lstm(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_layer_1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_gru(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(gru_units, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_layer_1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model