import os
import wget
import pandas as pd
import numpy as np
from zipfile import ZipFile
from keras.layers import Embedding
from ML_Pipeline.constants import *

def extractGlovefile():
    wget.download('http://nlp.stanford.edu/data/glove.6B.zip', out=GLOVE_DIR)
    with ZipFile(GLOVE_FILE_PATH, 'r') as zipObj:
        zipObj.extractall(GLOVE_DIR)


def read_glove_embeddings(glove_file_path=GLOVE_FILE_PATH):
    # Read GloVe embeddings into a DataFrame
    word_vec = pd.read_csv(glove_file_path, sep=r"\s", header=None, engine='python', encoding='iso-8859-1', on_bad_lines='skip')
    word_vec.set_index(0, inplace=True)
    return word_vec

def create_glove_embeddings(tokenizer):
    embeddings_index = read_glove_embeddings()
    embeddings_matrix = np.zeros((vocab_size, emb_dim))

    index_n_word = [(i, tokenizer.index_word[i]) for i in range(1, vocab_size) if tokenizer.index_word[i] in embeddings_index.index]
    idx, word = zip(*index_n_word)
    embeddings_matrix[list(idx), :] = embeddings_index.loc[list(word), :].values

    return embeddings_matrix


def build_embeddings(tokenizer):
    embeddings_matrix = create_glove_embeddings(tokenizer)
    embeddings_layer = Embedding(vocab_size, emb_dim, weights = [embeddings_matrix], input_length = max_length, trainable = False)
    return embeddings_layer