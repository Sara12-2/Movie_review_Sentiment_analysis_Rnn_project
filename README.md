# 📌 Sentiment Analysis with RNN (IMDB Movie Reviews)

This project uses a **Recurrent Neural Network (RNN)** to perform **sentiment analysis** on the IMDB dataset. The goal is to classify movie reviews as **positive** or **negative**.  

---

## 📂 Dataset
- **Source:** IMDB dataset (available in `tensorflow.keras.datasets.imdb`)  
- **Training samples:** 25,000  
- **Testing samples:** 25,000  
- **Labels:**  
  - `0` → Negative Review  
  - `1` → Positive Review  

Each review is encoded as a sequence of integers (word indices).  

---

## ⚙️ Preprocessing
- Only **top 10,000 most frequent words** are kept.  
- Each review is **padded/cut to 200 words** using `pad_sequences`.  

```python
max_features = 10000   # Vocabulary size
maxlen = 200           # Max review length
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
```
## 🏗 Model Architecture
```bash
model = models.Sequential([
    layers.Embedding(max_features, 128, input_length=maxlen),
    layers.SimpleRNN(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    layers.SimpleRNN(64, dropout=0.3, recurrent_dropout=0.3),
    layers.Dense(1, activation='sigmoid')
])


```

## 🔹 Explanation

Embedding Layer → converts word indices into 128-dim dense vectors.

RNN Layers (SimpleRNN) → capture sequence dependencies in the review text.

Dropout → prevent overfitting.

Dense Layer (sigmoid) → outputs probability (positive/negative sentiment).

## ⚡ Training

Optimizer: Adam (lr=0.0005)

Loss: Binary Crossentropy

Batch Size: 32

Epochs: up to 15 (with EarlyStopping)
```bash
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(x_train, y_train,
                    epochs=15,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop])
```
## 💾 Saving the Model
```bash
model.save("sentiment_rnn_improved.h5")

```
The trained model is saved as sentiment_rnn_improved.h5.

## 📊 Results

The model typically achieves 85%+ accuracy on IMDB test data.

Accuracy and loss can be visualized using matplotlib (optional).

## ▶️ Usage
```bash
1. Train Model
python train.py

2. Load & Predict
3. 
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("sentiment_rnn_improved.h5")

# Example prediction (dummy sequence of integers)
sample = np.random.randint(1, 10000, (1, 200))
prediction = model.predict(sample)
print("Positive" if prediction[0][0] > 0.5 else "Negative")
```
## 📝 Sample Reviews

Positive:

"I absolutely loved this movie. The acting was fantastic and the story kept me engaged."
Negative:

"This was a complete waste of time. The plot was boring and the acting was terrible."
