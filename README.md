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
* Libraries: Scikit-learn, Tensorflow, Keras, GloVe, nltk, pandas, numpy, Matplotlib

## How to use this project
1. __Clone the repo:__ `git clone https://github.com/jma52799/fake-jobs-ml.git`
2. __Install libraries:__ `pip install requirements.txt OR pip3 install requirements.txt`
3. __Run code:__ `python Engine.py OR python3 Engine.py (cd into the directory containing Engine.py first)`
4. Visit the ipynb notebook to see the results and execution of training the models from the beginning [Link to ipynb notebook](https://github.com/jma52799/fake-jobs-ml/blob/main/fakejobs.ipynb)

## Approach
### Data exploration and cleaning
Using the job listing data, I performed initial data exploration and cleaning to prepare it for further analysis and modeling. This involved examining the characteristics of each feature, dealing with missing data, and ensuring that the dataset was appropriately structured for training a classification model.

First, I analyzed the data by checking the number of records, features, and labels. Specifically, the dataset contains 17,880 job listings, with 18 different features, such as job title, description, company profile, and location. The primary target variable, fraudulent, indicates whether a job listing is legitimate (0) or fraudulent (1), which as shown below, revealed a significant class imbalance. To clean the dataset, I removed rows with missing values in essential fields and dropped columns that were irrelevant.

<img width="563" alt="Data Exploration" src="https://github.com/user-attachments/assets/07af8a89-7b7f-403f-8dbe-0d44877cf610">

### Data Preprocessing: NLTK processing
1. To begin, I combined multiple textual features into a single column,"text". Features like "description", "company_profile", and "requirements", etc., were merged into one text column. This allowed us to treat the job listing’s textual information as a single entity, which is crucial for NLP models.

2. Next, I applied various common NLTK preprocessing techniques This step (Lemmatization) involved:
- Converting text to lowercase 
- Removing URLs and special characters
- Stop word removal 
  
<img width="745" alt="NLTK preprocessing" src="https://github.com/user-attachments/assets/63e56319-2778-43a0-a3e6-a8d2027285cc">

### Sequence Data Preparation
1. Tokenization

A tokenizer was initialized and fitted to the training data to identify all the unique words. The total number of unique words came out to be 136,213. However such a large vocabulary increases the complexity and memory requirements of the model. So I analyzed the frequencies of the words to determine an optimal vocab size. The cumulative word frequency distribution plot as shown below revealed that a just a small subset of the words accounted for the majority of the total word occurences. The steep rise at the beginning of the curve shows that words with high frequencies take up a high percentage of the total word frequency. And after the steep rise, the tail (rest of the curve) flattens which means that a large proportion of words appear infrequently. After finding the optimal vocab size, a new tokenizer was initialized with the optimal vocab size and fitted to the training data. 

![Cumulative word frequency distribution](https://github.com/user-attachments/assets/3923d53a-ac59-45a5-9c82-813d8b074819)

2. Sequence data transformation

The "text" column was converted into sequences of tokens, where each word was converted to a corresponding token from the tokenizer's word index. I then padded the sequences to ensure that all input sequences had the same length, which helps the deep learning models to process the data consistently.

### Word Embeddings
For this project, I used pre-trained word embeddings to transform text data into meaningful numerical vectors that capture the semantic information of words. I chose the GloVe (Global Vectors for Word Representation) model developed by Stanford, and used the 100-dimensional version to balance performance and computational efficiency. I mapped the tokenizer's word index (created during tokenization) to the pre-trained GloVe embeddings, and built an embedding matrix that contained the GloVe vectors for the most frequent words in our dataset. This matrix was then used as the weight matrix for the embedding layer in our neural network model. I ensured the embedding layer was "non-trainable" so that the pre-trained GloVe embeddings would not be updated during models' training. 

### Build And Evaluate The Sequence Models
The sequence models trained are LSTM and GRU and both were build with the embedding layer, Dense (for classification), and Dropout (to prevent overfitting). I chose LSTM and GRU because both are good at learning and remembering long-term dependencies, which is ideal for long documents such as a job description. The first plot shows training and validation (test data) accuracy for the LSTM model and the second plot (below) shows the training and validation (test data) accuracy for the GRU model. Finally, a confusion matrix was generated to evaluate the performance of both models on the test dataset.

![LSTM](https://github.com/user-attachments/assets/73772d5d-6c39-4c30-8aa8-2eec45622380)
(LSTM - achieved accuracy slightly above 95%)

![GRU](https://github.com/user-attachments/assets/30551a4f-6091-4c39-be01-ddf497cc7c15)
(GRU - - achieved accuracy slightly above 95%)

### Saving The Models' Performances
The performance of the models are saved for future reference. 

![Performance Report](https://github.com/user-attachments/assets/d0e2d1de-0477-4142-a143-0868c807c93a)

## Summary and Future Works
Although both models achieved a high accuracy, using accuracy might not be the best measurement because the dataset is severely imablanced. As shown above, the recall is very low which suggests that the models are missing the vast majority of the fradulent job listings. A better analysis of the models would need to see the recall improve significantly. But it would require more amount of fradulent data to help balance the dataset. Or techniqus for balancing categorical data and data with extreme imbalance can be deploy to handle the extreme imbalance in this dataset.

## ML Pipeline
The *src* directory contains the *ML_pipeline* folder, which contains different files that are appropriately named after the step they perform towards creating the deep learning models.
