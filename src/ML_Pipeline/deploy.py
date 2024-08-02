
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from flask import Flask, jsonify
from pymongo import MongoClient, errors
from keras.models import model_from_json
from ML_Pipeline.utils import load_model
from ML_Pipeline import tokenizer
from ML_Pipeline.clean_data import clean_data
from ML_Pipeline.preprocess_data import preprocess_mongo_data
from ML_Pipeline.tokenizer import load_tokenizer, texts_to_sequences
from ML_Pipeline.constants import *
from ML_Pipeline.tokenizer import initialize_tokenizer, texts_to_sequences, pad_sequences

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# MongoDB connection details
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = 'test'
COLLECTION_NAME = 'jobs'

# Connect to MongoDB
client = None
collection = None
try:
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    # Attempt to retrieve server information to confirm the connection
    client.server_info()
except errors.ServerSelectionTimeoutError as err:
    # Handle the exception
    print(f"Failed to connect to MongoDB: {err}")
    collection = None

@app.route('/predict/<model_type>', methods=['GET'])
def predict(model_type):
    if collection is None:
        return jsonify({"error": "Failed to connect to MongoDB"}), 500

    if model_type not in ['lstm', 'gru']:
        return jsonify({"error": "Invalid model type. Use 'lstm' or 'gru'."}), 400
    
    # Load the specified model
    model = load_model(model_type, 1)
    
    # Fetch data from MongoDB
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    # Preprocess data
    df_preprocessed = preprocess_mongo_data(df)
    X = df_preprocessed['text'].values  # No label for prediction

    # Sequence Data Transformation
    tokenizer = load_tokenizer('../output/models/tokenizer.json')
    text_seq = texts_to_sequences(X, tokenizer)
    text_padded = pad_sequences(text_seq)

    # Make predictions
    predictions = model.predict(text_padded)
    predicted_labels = (predictions > 0.5).astype(int)

    # Identify fake rows
    fake_indices = np.where(predicted_labels == 1)[0]
    fake_rows = df.iloc[fake_indices]

    # Save the indices of fake rows
    fake_indices_list = fake_rows.index.tolist()
    with open(f'../output/fake_jobs/{model_type}_fake_indices.txt', 'w') as f:
        for idx in fake_indices_list:
            f.write(f"{idx}\n")

    return jsonify({"message": f"{model_type} prediction completed", "fake_indices": fake_indices_list})

@app.route('/delete/<model_type>', methods=['GET'])
def delete_fake_rows(model_type):
    if collection is None:
        return jsonify({"error": "Failed to connect to MongoDB"}), 500

    if model_type not in ['lstm', 'gru']:
        return jsonify({"error": "Invalid model type. Use 'lstm' or 'gru'."}), 400
    
    # Read fake indices from the appropriate file
    fake_indices_file = f'../output/fake_jobs/{model_type}_fake_indices.txt'
    if not os.path.exists(fake_indices_file):
        return jsonify({"error": f"Fake indices file for {model_type} predictions not found."}), 400
    
    with open(fake_indices_file, 'r') as f:
        fake_indices_list = [int(line.strip()) for line in f]

    # Fetch data from MongoDB
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Delete fake rows from the database
    for idx in fake_indices_list:
        collection.delete_one({'_id': df.iloc[idx]['_id']})

    return jsonify({"message": f"Fake rows deleted for {model_type}", "fake_indices_deleted": fake_indices_list})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)