import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords

wnl = nltk.stem.WordNetLemmatizer()
stopwords_dict = set(stopwords.words('english'))

def merge_text_columns(df, columns, col_name):
    df[col_name] = df[columns].apply(lambda x: ' '.join(x), axis=1)
    return df

# Stop words & lemmatization
def nltk_preprocessing(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)  # removing urls
    text = re.sub(r'#URL_[\w\d]+#', ' ', text)  # Remove strings that start with 'URL_' and are enclosed by '#'
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = [wnl.lemmatize(word) for word in text.split() if word not in stopwords_dict]
    #return ' '.join(text)
    return text

def preprocess_dataset(df, text_features, col_name, label):
    df = merge_text_columns(df, text_features, col_name)
    df[col_name] = df[col_name].apply(lambda x: nltk_preprocessing(x))
    X = df[col_name]
    y = df[label]
    return X, y

# Preprocess MongoDB data
def preprocess_mongo_data(df):
    df = df[['title', 'description']]
    df['text'] = df['title'] + ' ' + df['description']
    df['text'] = df['text'].apply(lambda x: nltk_preprocessing(x))
    return df[['text']]