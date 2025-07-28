import numpy as np
import pickle
import sys
import tensorflow as tf
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras.src.utils import pad_sequences
from tensorflow.keras.models import load_model

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.u = None
        self.b = None
        self.W = None

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output, alphas

with open("preprocessing.pkl", "rb") as f:
    data = pickle.load(f)
    tokenizer = data["tokenizer"]
    max_len = data["max_len"]
    stop_words = data["stop_words"]

model = load_model("sentiment_model.keras", custom_objects={"AttentionLayer": AttentionLayer})

def preprocess_text(text: str):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def predict_sentiment(text: str):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0]
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    predicted_class = np.argmax(pred)
    return label_map[predicted_class], float(pred[predicted_class])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for raw_text in sys.argv[1:]:
            sentiment, conf = predict_sentiment(raw_text)
            print(f"Text: {raw_text}\nSentiment: {sentiment} (Confidence: {conf:.2f})\n")
    else:
        print("Please provide input text(s) as command-line arguments.")
