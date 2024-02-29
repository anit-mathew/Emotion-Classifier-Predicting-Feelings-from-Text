import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('text.csv')

# Basic Statistics
print("Number of Instances:", len(data))
print("Number of Features:", len(data.columns))

# Class Distribution
class_distribution = data['label'].value_counts()
print("Class Distribution:")
print(class_distribution)

# Text Length Distribution
data['text_length'] = data['text'].apply(len)
print("Text Length Distribution:")
print(data['text_length'].describe())

# Word Frequency Analysis
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate a word cloud
all_text = ' '.join(data['text'])
wordcloud = WordCloud(width=800, height=400, random_state=42, background_color='white').generate(all_text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Text Data")
plt.show()

# Text Preprocessing Insights
print("Example of Lowercase Conversion:")
print(data['text'].head().apply(lambda x: x.lower()))
# (Add more preprocessing steps based on your needs)
