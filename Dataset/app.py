# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline

# # Load your pre-trained models and vectorizer
# lr_model = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/models/best_lr_model_afinn.pkl')
# count_vectorizer = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/feature_extract/lr_tfidf_afinn.pkl')
# lda_model = joblib.load('7 Topics Modelling/lda_models_fetract/models/lda_bow_with_balance.pkl')
# lda_vectorizer = joblib.load('7 Topics Modelling/lda_models_fetract/feature_extract/bow_vectorizer_with_balance.pkl')

# # Define a function to predict sentiment
# def predict_sentiment(headline):
#     # Transform headline using the count_vectorizer
#     transformed_headline = count_vectorizer.transform([headline])
#     # Predict sentiment using the logistic regression model
#     sentiment = lr_model.predict(transformed_headline)
#     return sentiment[0]

# # Define a function to get topic information
# def get_topics(article):
#     # Transform article using the lda_vectorizer
#     transformed_article = lda_vectorizer.transform([article])
#     # Get the topic distribution
#     topic_distribution = lda_model.transform(transformed_article)
    
#     # Colors for each topic
#     colors = plt.cm.tab10(np.linspace(0, 1, len(lda_model.components_)))
    
#     # Plot top words for each topic
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=False, sharey=False)
#     axes = axes.flatten()
#     feature_names = lda_vectorizer.get_feature_names_out()
    
#     for i, (topic, color) in enumerate(zip(lda_model.components_, colors)):
#         top_words_idx = np.argsort(topic)[::-1][:10]  # descending order
#         top_words = np.array(feature_names)[top_words_idx]
#         top_scores = topic[top_words_idx]

#         ax = axes[i]
#         ax.barh(top_words, top_scores, color=color)
#         ax.set_title(f'Topic {i}', fontsize=12, fontweight='bold')
#         ax.invert_yaxis()
#         ax.tick_params(axis='both', which='major', labelsize=10)
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     fig.suptitle('Top Words per Topic', fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save plot to BytesIO and display in Streamlit
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
#     return buf

# # Streamlit UI
# st.title('News Headline and Article Analysis')

# # User input for headline
# headline = st.text_input('Enter a news headline:')
# if headline:
#     sentiment = predict_sentiment(headline)
#     st.write(f'Sentiment: {sentiment}')

# # User input for article
# article = st.text_area('Enter a news article text:')
# if article:
#     topics_plot = get_topics(article)
#     st.image(topics_plot)


# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline

# # Load your pre-trained models and vectorizer
# lr_model = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/models/best_lr_model_afinn.pkl')
# count_vectorizer = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/feature_extract/lr_tfidf_afinn.pkl')
# lda_model = joblib.load('7 Topics Modelling/lda_models_fetract/models/lda_bow_with_balance.pkl')
# lda_vectorizer = joblib.load('7 Topics Modelling/lda_models_fetract/feature_extract/bow_vectorizer_with_balance.pkl')

# # Define a function to predict sentiment
# def predict_sentiment(headline):
#     # Transform headline using the count_vectorizer
#     transformed_headline = count_vectorizer.transform([headline])
#     # Predict sentiment using the logistic regression model
#     sentiment = lr_model.predict(transformed_headline)
#     return sentiment[0]

# # Define a function to get topic information
# def get_topics(article):
#     # Transform article using the lda_vectorizer
#     transformed_article = lda_vectorizer.transform([article])
#     # Get the topic distribution
#     topic_distribution = lda_model.transform(transformed_article)[0]
    
#     # Get top topics based on distribution
#     top_topic_indices = topic_distribution.argsort()[-5:][::-1]  # Get indices of top 5 topics
#     top_topic_probabilities = topic_distribution[top_topic_indices]
    
#     # Colors for each topic
#     colors = plt.cm.tab10(np.linspace(0, 1, len(lda_model.components_)))
    
#     # Plot top words for each topic
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=False, sharey=False)
#     axes = axes.flatten()
#     feature_names = lda_vectorizer.get_feature_names_out()
    
#     for i, (topic, color) in enumerate(zip(lda_model.components_, colors)):
#         top_words_idx = np.argsort(topic)[::-1][:10]  # descending order
#         top_words = np.array(feature_names)[top_words_idx]
#         top_scores = topic[top_words_idx]

#         ax = axes[i]
#         ax.barh(top_words, top_scores, color=color)
#         ax.set_title(f'Topic {i}', fontsize=12, fontweight='bold')
#         ax.invert_yaxis()
#         ax.tick_params(axis='both', which='major', labelsize=10)
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     fig.suptitle('Top Words per Topic', fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save plot to BytesIO and display in Streamlit
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
    
#     return top_topic_indices, top_topic_probabilities, buf

# # Streamlit UI
# st.title('News Headline and Article Analysis')

# # User input for headline
# headline = st.text_input('Enter a news headline:')
# if headline:
#     sentiment = predict_sentiment(headline)
#     st.write(f'Sentiment: {sentiment}')

# # User input for article
# article = st.text_area('Enter a news article text:')
# if article:
#     top_topic_indices, top_topic_probabilities, topics_plot = get_topics(article)
#     st.image(topics_plot)
    
#     # Display top topics
#     st.write('Top Topics Based on Article:')
#     for index, probability in zip(top_topic_indices, top_topic_probabilities):
#         st.write(f'Topic {index}: {probability:.2f}')


# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from nltk.corpus import stopwords, words
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import contractions
# import re

# # Load your pre-trained models and vectorizer
# lr_model = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/models/best_lr_model_afinn.pkl')
# count_vectorizer = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/feature_extract/lr_tfidf_afinn.pkl')
# lda_model = joblib.load('7 Topics Modelling/lda_models_fetract/models/lda_bow_with_balance.pkl')
# lda_vectorizer = joblib.load('7 Topics Modelling/lda_models_fetract/feature_extract/bow_vectorizer_with_balance.pkl')

# # Text cleaning functions
# def lowercase_text(text):
#     return text.lower()

# def expand_contractions(text):
#     return contractions.fix(text)

# def remove_cnn(text):
#     return text.replace('cnn', '')

# def remove_short_words(text):
#     return ' '.join([word for word in text.split() if len(word) > 2])

# def remove_symbols(text):
#     symbol_pattern = re.compile(r'[\(\)\[\]:]')
#     return symbol_pattern.sub('', text)

# def remove_symbols_digits(text):
#     return re.sub('[^a-zA-Z\s]', ' ', text)

# def remove_urls(text):
#     return re.sub(r'http\S+', '', text)

# def remove_html_tags(text):
#     return re.sub(r'<[^>]+>', '', text)

# def remove_whitespace(text):
#     return ' '.join(text.split())

# def remove_punctuation(text):
#     return re.sub(r'[^\w\s]', '', text)

# def remove_stopwords(text):
#     stop_words = set(stopwords.words('english'))
#     return ' '.join([token for token in text.split() if token.lower() not in stop_words])

# def lemmatize_text(text):
#     lemmatizer = WordNetLemmatizer()
#     return ' '.join([lemmatizer.lemmatize(token) for token in text.split()])

# def tokenize_text(text):
#     tokens = word_tokenize(text)
#     return tokens

# english_words = set(words.words())
# def remove_non_english(tokens):
#     english_tokens = [word for word in tokens if word in english_words]
#     return ' '.join(english_tokens)

# # Apply all cleaning functions
# def clean_text(text):
#     text = lowercase_text(text)
#     text = expand_contractions(text)
#     text = remove_cnn(text)
#     text = remove_short_words(text)
#     text = remove_symbols(text)
#     text = remove_symbols_digits(text)
#     text = remove_urls(text)
#     text = remove_html_tags(text)
#     text = remove_whitespace(text)
#     text = remove_punctuation(text)
#     text = remove_stopwords(text)
#     text = lemmatize_text(text)
#     tokens = tokenize_text(text)
#     text = remove_non_english(tokens)
#     return text

# # Define a function to predict sentiment
# def predict_sentiment(headline):
#     cleaned_headline = clean_text(headline)
#     transformed_headline = count_vectorizer.transform([cleaned_headline])
#     sentiment = lr_model.predict(transformed_headline)
#     return sentiment[0]

# # Define a function to get topic information
# def get_topics(article):
#     cleaned_article = clean_text(article)
#     transformed_article = lda_vectorizer.transform([cleaned_article])
#     topic_distribution = lda_model.transform(transformed_article)[0]
    
#     top_topic_indices = topic_distribution.argsort()[-5:][::-1]  # Get indices of top 5 topics
#     top_topic_probabilities = topic_distribution[top_topic_indices]
    
#     colors = plt.cm.tab10(np.linspace(0, 1, len(lda_model.components_)))
    
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=False, sharey=False)
#     axes = axes.flatten()
#     feature_names = lda_vectorizer.get_feature_names_out()
    
#     for i, (topic, color) in enumerate(zip(lda_model.components_, colors)):
#         top_words_idx = np.argsort(topic)[::-1][:10]  # descending order
#         top_words = np.array(feature_names)[top_words_idx]
#         top_scores = topic[top_words_idx]

#         ax = axes[i]
#         ax.barh(top_words, top_scores, color=color)
#         ax.set_title(f'Topic {i}', fontsize=12, fontweight='bold')
#         ax.invert_yaxis()
#         ax.tick_params(axis='both', which='major', labelsize=10)
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     fig.suptitle('Top Words per Topic', fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
    
#     return top_topic_indices, top_topic_probabilities, buf

# # Streamlit UI
# st.title('News Headline and Article Analysis')

# # User input for headline
# headline = st.text_input('Enter a news headline:')
# if headline:
#     sentiment = predict_sentiment(headline)
#     st.write(f'Sentiment: {sentiment}')

# # User input for article
# article = st.text_area('Enter a news article text:')
# if article:
#     top_topic_indices, top_topic_probabilities, topics_plot = get_topics(article)
#     st.image(topics_plot)
    
#     st.write('Top Topics Based on Article:')
#     for index, probability in zip(top_topic_indices, top_topic_probabilities):
#         st.write(f'Topic {index}: {probability:.2f}')

# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from nltk.corpus import stopwords, words
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import contractions
# import re

# # Load your pre-trained models and vectorizer
# lr_model = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/models/best_lr_model_afinn.pkl')
# count_vectorizer = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/feature_extract/lr_tfidf_afinn.pkl')
# lda_model = joblib.load('7 Topics Modelling/lda_models_fetract/models/lda_bow_with_balance.pkl')
# lda_vectorizer = joblib.load('7 Topics Modelling/lda_models_fetract/feature_extract/bow_vectorizer_with_balance.pkl')

# # Text cleaning functions
# def lowercase_text(text):
#     return text.lower()

# def expand_contractions(text):
#     return contractions.fix(text)

# def remove_cnn(text):
#     return text.replace('cnn', '')

# def remove_short_words(text):
#     return ' '.join([word for word in text.split() if len(word) > 2])

# def remove_symbols(text):
#     symbol_pattern = re.compile(r'[\(\)\[\]:]')
#     return symbol_pattern.sub('', text)

# def remove_symbols_digits(text):
#     return re.sub('[^a-zA-Z\s]', ' ', text)

# def remove_urls(text):
#     return re.sub(r'http\S+', '', text)

# def remove_html_tags(text):
#     return re.sub(r'<[^>]+>', '', text)

# def remove_whitespace(text):
#     return ' '.join(text.split())

# def remove_punctuation(text):
#     return re.sub(r'[^\w\s]', '', text)

# def remove_stopwords(text):
#     stop_words = set(stopwords.words('english'))
#     return ' '.join([token for token in text.split() if token.lower() not in stop_words])

# def lemmatize_text(text):
#     lemmatizer = WordNetLemmatizer()
#     return ' '.join([lemmatizer.lemmatize(token) for token in text.split()])

# def tokenize_text(text):
#     tokens = word_tokenize(text)
#     return tokens

# english_words = set(words.words())
# def remove_non_english(tokens):
#     english_tokens = [word for word in tokens if word in english_words]
#     return ' '.join(english_tokens)

# # Apply all cleaning functions
# def clean_text(text):
#     text = lowercase_text(text)
#     text = expand_contractions(text)
#     text = remove_cnn(text)
#     text = remove_short_words(text)
#     text = remove_symbols(text)
#     text = remove_symbols_digits(text)
#     text = remove_urls(text)
#     text = remove_html_tags(text)
#     text = remove_whitespace(text)
#     text = remove_punctuation(text)
#     text = remove_stopwords(text)
#     text = lemmatize_text(text)
#     tokens = tokenize_text(text)
#     text = remove_non_english(tokens)
#     return text

# # Define a function to predict sentiment
# def predict_sentiment(headline):
#     cleaned_headline = clean_text(headline)
#     transformed_headline = count_vectorizer.transform([cleaned_headline])
#     sentiment = lr_model.predict(transformed_headline)
#     sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
#     return sentiment_mapping[sentiment[0]]

# # Define a function to get topic information
# def get_topics(article):
#     cleaned_article = clean_text(article)
#     transformed_article = lda_vectorizer.transform([cleaned_article])
#     topic_distribution = lda_model.transform(transformed_article)[0]
    
#     top_topic_indices = topic_distribution.argsort()[-5:][::-1]  # Get indices of top 5 topics
#     top_topic_probabilities = topic_distribution[top_topic_indices]
    
#     colors = plt.cm.tab10(np.linspace(0, 1, len(lda_model.components_)))
    
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=False, sharey=False)
#     axes = axes.flatten()
#     feature_names = lda_vectorizer.get_feature_names_out()
    
#     for i, (topic, color) in enumerate(zip(lda_model.components_, colors)):
#         top_words_idx = np.argsort(topic)[::-1][:10]  # descending order
#         top_words = np.array(feature_names)[top_words_idx]
#         top_scores = topic[top_words_idx]

#         ax = axes[i]
#         ax.barh(top_words, top_scores, color=color)
#         ax.set_title(f'Topic {i}', fontsize=12, fontweight='bold')
#         ax.invert_yaxis()
#         ax.tick_params(axis='both', which='major', labelsize=10)
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     fig.suptitle('Top Words per Topic', fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
    
#     return top_topic_indices, top_topic_probabilities, buf

# # Streamlit UI
# st.title('News Headline and Article Analysis')

# # User input for headline
# headline = st.text_input('Enter a news headline:')
# if headline:
#     sentiment = predict_sentiment(headline)
#     st.write(f'Sentiment: {sentiment}')

# # User input for article
# article = st.text_area('Enter a news article text:')
# if article:
#     top_topic_indices, top_topic_probabilities, topics_plot = get_topics(article)
#     st.image(topics_plot)
    
#     st.write('Top Topics Based on Article:')
#     for index, probability in zip(top_topic_indices, top_topic_probabilities):
#         st.write(f'Topic {index}: {probability:.2f}')

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import re

# Load your pre-trained models and vectorizer
lr_model = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/models/best_lr_model_afinn.pkl')
count_vectorizer = joblib.load('5 Unsupervised Sentiment Analysis/lexicon_models_fextract/feature_extract/lr_tfidf_afinn.pkl')
lda_model = joblib.load('7 Topics Modelling/lda_models_fetract/models/lda_bow_with_balance.pkl')
lda_vectorizer = joblib.load('7 Topics Modelling/lda_models_fetract/feature_extract/bow_vectorizer_with_balance.pkl')

# Text cleaning functions
def lowercase_text(text):
    return text.lower()

def expand_contractions(text):
    return contractions.fix(text)

def remove_cnn(text):
    return text.replace('cnn', '')

def remove_short_words(text):
    return ' '.join([word for word in text.split() if len(word) > 2])

def remove_symbols(text):
    symbol_pattern = re.compile(r'[\(\)\[\]:]')
    return symbol_pattern.sub('', text)

def remove_symbols_digits(text):
    return re.sub('[^a-zA-Z\s]', ' ', text)

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

def remove_whitespace(text):
    return ' '.join(text.split())

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([token for token in text.split() if token.lower() not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in text.split()])

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

english_words = set(words.words())
def remove_non_english(tokens):
    english_tokens = [word for word in tokens if word in english_words]
    return ' '.join(english_tokens)

# Apply all cleaning functions
def clean_text(text):
    text = lowercase_text(text)
    text = expand_contractions(text)
    text = remove_cnn(text)
    text = remove_short_words(text)
    text = remove_symbols(text)
    text = remove_symbols_digits(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_whitespace(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    tokens = tokenize_text(text)
    text = remove_non_english(tokens)
    return text

# Define a function to predict sentiment
def predict_sentiment(headline):
    cleaned_headline = clean_text(headline)
    transformed_headline = count_vectorizer.transform([cleaned_headline])
    sentiment = lr_model.predict(transformed_headline)
    sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    return sentiment_mapping[sentiment[0]]

# Define a function to get topic information
def get_topics(article):
    cleaned_article = clean_text(article)
    transformed_article = lda_vectorizer.transform([cleaned_article])
    topic_distribution = lda_model.transform(transformed_article)[0]
    
    top_topic_indices = topic_distribution.argsort()[-5:][::-1]  # Get indices of top 5 topics
    top_topic_probabilities = topic_distribution[top_topic_indices]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(lda_model.components_)))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharex=False, sharey=False)
    axes = axes.flatten()
    feature_names = lda_vectorizer.get_feature_names_out()
    
    for i, (topic, color) in enumerate(zip(lda_model.components_, colors)):
        top_words_idx = np.argsort(topic)[::-1][:10]  # descending order
        top_words = np.array(feature_names)[top_words_idx]
        top_scores = topic[top_words_idx]

        ax = axes[i]
        ax.barh(top_words, top_scores, color=color)
        ax.set_title(f'Topic {i}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('Top Words per Topic', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return top_topic_indices, top_topic_probabilities, buf

# Streamlit UI
st.title('Sentiment Analysis and Topic Modelling on News Headline and Article to Detect Trends')

col1, col2 = st.columns([3, 1])

with col1:
    # User input for headline
    headline = st.text_input('Enter a news headline:', '')
    article = st.text_area('Enter a news article text:', '', height=300)

with col2:
    # Display sentiment next to headline input
    if headline:
        sentiment = predict_sentiment(headline)
        st.write(f'Sentiment: {sentiment}')
        
    # Display topic word score chart next to article input
    if article:
        top_topic_indices, top_topic_probabilities, topics_plot = get_topics(article)
        st.image(topics_plot, use_column_width=True)
        
        st.write('Top Topics Based on Article:')
        for index, probability in zip(top_topic_indices, top_topic_probabilities):
            st.write(f'Topic {index}: {probability:.2f}')













