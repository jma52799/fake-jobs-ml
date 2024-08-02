#--- Clean Data Constants ---#
remove_columns = ['job_id', 'location']
label = ['fraudelent']
text_features = ['title', 'company_profile', 'description', 'requirements', 'benefits']

#--- Preprocess Data Constants ---#
col_name = 'text'

#--- Tokenization Constants ---#
vocab_size = 20000
max_length = 400        #100
oov_token = '<OOV>'
padding_type = 'post'
trunc_type = 'post'

#--- Embedding Constants ---#
emb_dim = 100
GLOVE_DIR = "../input/glove/"
GLOVE_FILE_PATH = GLOVE_DIR + "glove.6B." + str(emb_dim) + "d.txt"

#--- Model Constants ---#
lstm_units = 128
gru_units = 128
hidden_layer_1 = 32
epochs = 20
batch_size = 256
classifier = 'binary'