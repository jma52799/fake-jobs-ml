import json
import io
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import pad_sequences
from ML_Pipeline.constants import *

def initialize_tokenizer(df_train):
    tokenizer = Tokenizer(oov_token = oov_token, num_words = vocab_size)
    tokenizer.fit_on_texts(df_train)
    return tokenizer

def texts_to_sequences(texts, tokenizer):
    text_sequences = tokenizer.texts_to_sequences(texts)
    return text_sequences

def pad_sequences_data(text_sequences):
    padded_sequences = pad_sequences(text_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    return padded_sequences

def save_tokenizer(tokenizer, num_words = vocab_size, output_dir='../output/models/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = output_dir + 'tokenizer_' + str(num_words) + '.json'
    
    with io.open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    outfile.close()
    

def load_tokenizer(filepath):
    with open(filepath, 'r', encoding='utf-8') as json_file:
        tokenizer_json = json.load(json_file)
        tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer