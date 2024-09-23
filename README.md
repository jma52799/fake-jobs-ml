# Fake jobs classification

## Project Overview

*What is fake job postings?*\
Fake job postings are job posts from companies for openings that don't actually exist. These fake job posts look realistic and can complicate the already complicated search process for job seekers.

*How big is this problem?*\
Fake job posts are proliferating online. According to a survey of 650 hiring managers conducted by the career site Resume Builder, 40% of the companies posted at least one fake job listing in the year. Furthermore, certain job sites contain more fradulent job posts than other. For example, an article published by Business Insider stated that Indeed accounts for 32% of fraudulent job listings while LinkedIn contributed to 7% of the listings.

*Solution*\
Online job posts have a uniform data format: text. By using NLP techniques and Deep Learning sequence to sequence techniques, this project aims to identify the deceiving listings from the real ones.

## Dataset 
* __Source:__ [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

## Tech Stack
* Language: Python
* Libraries: Scikit-learn, Tensorflow, Keras, GloVe, nltk, pandas, numpy

## How to use this project
1. __Clone the repo:__ `git clone https://github.com/jma52799/fake-jobs-ml.git`
2. __Install libraries:__ `pip install requirements.txt OR pip3 install requirements.txt`
3. __Run code:__ `python Engine.py OR python3 Engine.py (cd into the directory containing Engine.py first)`
4. Visit the ipynb notebook to see the results and execution of training the models from the beginning [Link to ipynb notebook](https://github.com/jma52799/fake-jobs-ml/blob/main/fakejobs.ipynb)

## Approach
### Data exploration and cleaning
Using the job listing data, I performed initial data exploration and cleaning to prepare it for further analysis and modeling. This involved examining the characteristics of each feature, dealing with missing data, and ensuring that the dataset was appropriately structured for training a classification model.

First, I analyzed the data by checking the number of records, features, and labels. Specifically, the dataset contains 17,880 job listings, with 18 different features, such as job title, description, company profile, and location. The primary target variable, fraudulent, indicates whether a job listing is legitimate (0) or fraudulent (1). However, during our exploration, we discovered significant class imbalance. To clean the dataset, I removed rows with missing values in essential fields and dropped columns that were irrelevant.

<img width="563" alt="Data Exploration" src="https://github.com/user-attachments/assets/07af8a89-7b7f-403f-8dbe-0d44877cf610">

### Data Preprocessing: NLTK processing
1. To begin, I combined multiple textual features into a single column,"text". Features like "description", "company_profile", and "requirements", etc., were merged into one text column. This allowed us to treat the job listing’s textual information as a single entity, which is crucial for NLP models.

2. Next, I applied various common NLTK preprocessing techniques This step involved:
- Converting text to lowercase 
- Removing URLs and special characters
- Stop word removal 
- Lemmatization

<img width="745" alt="NLTK preprocessing" src="https://github.com/user-attachments/assets/63e56319-2778-43a0-a3e6-a8d2027285cc">

### Sequence Data Preparation
1. Tokenized the text after preprocessing
  - Found the optimal vocab size
  - Initialized and fitted the tokenizer with the determined vocab size
2. Sequence data transformation
  - Converted text to sequences
  - Padded the sequences to uniform length

### Word Embeddings
1. Downloaded the pre-trained word vectors (GloVe from Standford)
2. Used GloVe to convert text into meaningful numberical vectors

### Build Sequence Models
The sequence models trained are LSTM and GRU and both were build with the embedding layer, Dense, and Dropout

### Validate trained model
The performance of the model training is evaluated based on accuracy. And a confusion matrix is used to evaluate the performance of both models on the test dataset.

## ML Pipeline
The *src* directory contains the *ML_pipeline* folder, which contains different files that are appropriately named after the step they perform towards creating the deep learning models.
