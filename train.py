import numpy as np
import pandas as pd
import nltk
import pickle
import kagglehub
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Lambda, Reshape
from tensorflow.keras.models import Model
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras.src.layers import Embedding
from keras.src.utils import pad_sequences
from keras.src.legacy.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils.np_utils import to_categorical

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

train = pd.read_csv(path+'/train.csv', encoding='latin1')
test = pd.read_csv(path+'/test.csv', encoding='latin1')

def remove_stopwords(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

train = train.dropna()

X_train = train[['text']]
y_train = train['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})

test = test.dropna()
X_test = test[['text']]
y_test = test['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})

# Clean and preprocess the text
train_texts = X_train['text'].astype(str).apply(remove_stopwords)
test_texts = X_test['text'].astype(str).apply(remove_stopwords)

y_train = to_categorical(y_train.astype(int).tolist(), num_classes=3)
y_test = to_categorical(y_test.astype(int).tolist(), num_classes=3)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_seq = tokenizer.texts_to_sequences(train_texts)
test_seq = tokenizer.texts_to_sequences(test_texts)

max_len = max(max(len(seq) for seq in train_seq), 50)

X_train_pad = pad_sequences(train_seq, max_len)
X_test_pad = pad_sequences(test_seq, max_len)

word_idx = tokenizer.word_index

# print first 5 keys
for key in list(word_idx.keys())[:5]:
    print(key, word_idx[key])

path = kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300")
path += '/GoogleNews-vectors-negative300.bin'
print("Path to dataset files:", path)

word2vec = KeyedVectors.load_word2vec_format(path, binary=True)

embedding_dim = 300
vocab_size = len(word_idx)+1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_idx.items():
    if word in word2vec:
        embedding_matrix[i] = word2vec[word]

# Define the attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.u = None
        self.b = None
        self.W = None

    def build(self, input_shape):
        """Trainable weights for attention mechanism"""
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        # Score computation
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)

        alphas = tf.nn.softmax(vu)

        # weighted sum of input
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output, alphas


# Sample Bi-LSTM model with Attention
def create_model(input_shape):
    inputs = Input(shape=input_shape)

    # Embeddings Layer
    embedding_layer = Embedding(
        input_dim = vocab_size,
        output_dim = embedding_dim,
        input_length = max_len,
        trainable=False)(inputs)

    # Bi LSTM layer
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)

    # Add Attention Layer
    attention_out, attention_weights = AttentionLayer()(lstm_out)

    # Reshape Layer
    reshaped = Reshape((1, 128))(attention_out)

    # LSTM layer post attention
    lstm_after_attn = LSTM(64, return_sequences=False)(reshaped)

    # Dense layer after LSTM
    dense = Dense(128, activation='relu')(lstm_after_attn)

    # Output layer with softmax activation for 3-class classification
    outputs = Dense(3, activation='softmax')(lstm_after_attn)

    # Return the model
    return Model(inputs, outputs)


# Set input shapes and compile the model
input_shape = (50,)

model = create_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Train Model
model.fit(X_train_pad, np.array(y_train), epochs=50, batch_size=32, validation_split=0.2)


# Test Model
loss, accuracy = model.evaluate(X_test_pad, np.array(y_test))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("sentiment_model.keras")
