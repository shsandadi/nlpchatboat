#Import necessary libraries
import pandas as pd
import numpy as np
pd.set_option('max_colwidth', None)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import re
import string
import unidecode
from autocorrect import Speller
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
nlp = spacy.load('en_core_web_sm')  # Loading the envrionment config
from spacy.lang.en import English
en_nlp = English()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from bs4 import BeautifulSoup
import gensim.downloader as api
import pickle
import random
import warnings
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,TimeDistributed,Flatten,Input,Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
warnings.filterwarnings('ignore')
print("TensorFlow version:", tf.__version__)
import streamlit as st
import pickle
import tensorflow as tf

def load_model():
    model_path = r'nlp_chatbot.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    st.set_page_config(page_title="NLP Chatbox for Accident", layout="centered")
    st.title("NLP Chatbox for Accident")
    st.markdown('<style>h1{color: #336699;}</style>', unsafe_allow_html=True)
   # nlp_model = load_model()
    tf_version = tf.__version__
    st.sidebar.header(f"Information (TensorFlow Version: {tf_version})")
    st.sidebar.markdown("This is a simple NLP chatbox for handling accident-related queries.")
    user_input = st.text_input("User Input:", "")
    st.text_area("Chatbot Response:", height=100, max_chars=500, value="", key="chat_response", disabled=True)
    if st.button("Submit"):
        #response = process_user_input(user_input, nlp_model)
        nlp_model='abc'
        response = process_user_input(user_input, nlp_model)
        st.text_area("Chatbot Response:", value=response, key="chat_response", disabled=True, background="#F0F8FF", font_color="#333", font_family="Arial, sans-serif")

def process_user_input(user_input, model):
    return "Response from your model here."

if __name__ == "__main__":
    main()
