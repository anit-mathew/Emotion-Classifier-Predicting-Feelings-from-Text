import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np


# Load data
data = pd.read_csv('text.csv')

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Tokenize text
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences for equal length
max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=16, input_length=max_len))
model.add(LSTM(32))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test))

# Evaluate the model
y_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_prob, axis=1)

# Print classification report
print(classification_report(y_test, y_pred))

def predict_emotion(model, tokenizer, max_len, user_input):
    # Tokenize and pad the user input
    user_input_seq = tokenizer.texts_to_sequences([user_input])
    user_input_pad = pad_sequences(user_input_seq, maxlen=max_len, padding='post')

    # Predict the emotion
    emotion_prob = model.predict(user_input_pad)[0]
    predicted_emotion = label_encoder.classes_[np.argmax(emotion_prob)]

    return predicted_emotion

while True:
    user_input = input("Enter your text (or type 'no' to exit): ")
    
    if user_input.lower() == 'no':
        print("Exiting the program.")
        break
    
    predicted_feeling = predict_emotion(model, tokenizer, max_len, user_input)
    print("Predicted Feeling:", predicted_feeling)
