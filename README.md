# Sentiment Analysis using BiLSTM with Attention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamadbijarani/Sentiment-Analysis-using-LSTM/blob/main/Sentiment_analysis.ipynb)

This project implements a Sentiment Analysis pipeline using a Bi-directional LSTM (BiLSTM) model with an attention mechanism. It leverages pre-trained GloVe word embeddings and TensorFlow/Keras to classify the sentiment of text (e.g., movie reviews) as **positive** or **negative**.

## 🚀 Project Highlights

- Preprocessing using NLTK and custom cleaning techniques
- Word vectorization with pre-trained **GloVe embeddings**
- Deep learning model built with:
  - **Embedding Layer** (non-trainable)
  - **Bidirectional LSTM**
  - **Attention Mechanism**
  - **Dense Output Layer**
- Achieves competitive accuracy on binary sentiment classification tasks
- Includes training, prediction, and model serialization (using `pickle`)

---

## 📊 Dataset

This code is originally trained on tweets dataset [here](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset). This project would support any labeled sentiment dataset (like IMDB, SST, or custom CSVs) after some changes. The sample code is configured to work with a CSV file containing:

| Text                      | Sentiment |
| ------------------------- | --------- |
| "I loved this movie!"     | Positive  |
| "It was boring and slow." | Negative  |

---

## 📌 Dependencies

* Python 3.8+
* TensorFlow / Keras
* NLTK
* NumPy / Pandas
* Gensim (for GloVe)
* Pickle

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/hamadbijarani/Sentiment-Analysis-using-LSTM.git
cd Sentiment-Analysis-using-LSTM

# Create environment (recommended)
conda create -n sentiment-env python=3.9
conda activate sentiment-env

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Model Architecture

* **Input**: Padded sequences of tokenized text
* **Embedding**: GloVe word vectors (100D or 300D)
* **BiLSTM**: Captures forward and backward context
* **Attention**: Learns to focus on key tokens
* **Dense Output**: Softmax or Sigmoid for classification

---

## 📈 Training

```bash
python train.py
```

Trains the model using the dataset and saves the tokenizer, model variables, and model.

---

## ✅ Predicting

```bash
python test.py "...some string"
```

Predicts sentiment on input given by the user as arguments.

---

## 🧪 Inference

Once trained, you can load the model and tokenizer using `pickle` or `tensorflow.keras.models.load_model()` to make predictions on new text samples.

---

## 🔍 Example Prediction

```python
from test import predict_sentiment

sentiment, conf = predict_sentiment("I really enjoyed this movie!")
print(f"Sentiment: {sentiment} (Confidence: {conf})")
# Output: Positive, 
```


---

## 📜 License

This project is open-source and available under the MIT License.

