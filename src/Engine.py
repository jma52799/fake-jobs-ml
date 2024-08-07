import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from ML_Pipeline.constants import *
from ML_Pipeline.clean_data import clean_data
from ML_Pipeline.embedding import build_embeddings
from ML_Pipeline.evaluate_model import model_evaluation, plot_training_history, performance_report
from ML_Pipeline.preprocess_data import preprocess_dataset
from ML_Pipeline.store_model import save_model_to_file
from ML_Pipeline.tokenizer import initialize_tokenizer, save_tokenizer, texts_to_sequences, pad_sequences
from ML_Pipeline.build_models import build_lstm, build_gru

val = int(input('Train - 0\nDevelopment Deployment - 1\n'))
if val == 0:
    df = pd.read_csv('../input/job_postings.csv', on_bad_lines='skip')

    # Clean data
    df = clean_data(df, remove_columns)

    # Preprocess data
    #   1. Merge text columns
    #   2. Clean text data 
    #   3. Nltk processing (stop words & lemmatization)
    print('Preprocessing data...')
    X, y = preprocess_dataset(df, text_features, col_name, label)

    # Split data into train and test set
    print('Splitting data into train and test set...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Sequence Data Transformation
    #   1. Initialize and fit tokenizer
    #   2. Convert list of text to list of sequences (list of integers)
    #   3. Pad sequence to ensure uniform length
    print('Fitting tokenizer...')
    tokenizer = initialize_tokenizer(X_train)
    save_tokenizer(tokenizer)

    print('Converting text to sequence...')
    train_text_seq = tokenizer.texts_to_sequences(X_train, tokenizer)
    test_text_seq = tokenizer.texts_to_sequences(X_test, tokenizer)

    print('Padding sequence...')
    train_text_padded = pad_sequences(train_text_seq)
    test_text_padded = pad_sequences(test_text_seq)

    # Feature Engineering (Embedding)
    print('Building embedding layer...')
    embedding_layer = build_embeddings(tokenizer)

    # Model buiding, training, evaluating, and prediction
    print('Training LSTM model')
    model_lstm = build_lstm(embedding_layer)
    history = model_lstm.fit(train_text_padded, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    plot_training_history(history)
    save_model_to_file(model_lstm, 'lstm', 1)
    #accuracy_score = model_evaluation(model_lstm, test_text_padded, y_test)
    performance_report(model_lstm, test_text_padded, y_test, 'lstm', 'train') 

    print('Training GRU model')
    model_gru = build_gru(embedding_layer)
    history = model_gru.fit(train_text_padded, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    plot_training_history(history)
    save_model_to_file(model_gru, 'gru', 1)
    #accuracy_score = model_evaluation(model_gru, test_text_padded, y_test)
    performance_report(model_gru, test_text_padded, y_test, 'gru', 'train')
else:
    process = subprocess.Popen(['python3', 'deployment.py'], 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               )
    
    for stdout_line in process.stdout:
        print(stdout_line.decode('utf-8'))

    for stderr_line in process.stderr:
        print(stderr_line.decode('utf-8'))
    
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)