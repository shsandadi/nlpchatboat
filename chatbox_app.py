import pickle
import streamlit as st
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
import tensorflow as tf
import pandas as pd
import re
import nltk
import string
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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import requests, zipfile, io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import unicodedata


def load_model():
    model_path = r'nlp_chatbot.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
@st.cache_data
def extractGlove(url):
    response = requests.get(url)
    response.raise_for_status()
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall('.') 
@st.cache_data
def embedded_words():
     embeddings_index = {}
     EMBEDDING_FILE = './glove.6B.200d.txt'
     f = open(EMBEDDING_FILE)
     for line in tqdm(f):
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
     f.close()
     return embeddings_index
def main():
    st.set_page_config(page_title="NLP Chatbox for Accident", page_icon="🚑", layout="centered")
    st.title("NLP Chatbot for Accident")
    st.markdown('<style>h1{color: #336699;}</style>', unsafe_allow_html=True)
    nlp_model = load_model()
    tf_version = tf.__version__
    st.sidebar.header(f"This is chatbot of Group 6 to predict the accident level")
    st.sidebar.markdown("This is a simple NLP chatbot for handling accident-related queries.")
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    extractGlove(url)
    embedded_words()
    # User inputs for each column
    data_input = st.text_input("Date:")
    countries_input=st.selectbox("Countries:", ('Country_01', 'Country_02', 'Country_03'))
    #countries_input = st.text_input("Countries:")
    local_input=st.selectbox("Local:", ('Local_01', 'Local_02', 'Local_03','Local_04', 'Local_05', 'Local_06','Local_07', 'Local_08', 'Local_09','Local_10', 'Local_11', 'Local_12'))
    #local_input = st.text_input("Local:")
    #industry_sector_input = st.text_input("Industry Sector:")
    industry_sector_input=st.selectbox("Industry Sector:", ('Mining', 'Metals', 'Others'))
    potential_accident_level_input=st.selectbox("Potential Accident Level:", ('I', 'II', 'III','IV','V','VI'))
    #potential_accident_level_input = st.text_input("Potential Accident Level:")
    genre_input=st.selectbox("Genre:", ('Male', 'Female'))
    #genre_input = st.text_input("Genre:")
    employee_third_party_input = st.text_input("Employee or Third Party:")
    critical_risk_input = st.text_input("Critical Risk:")
    description_input = st.text_area("Description:", height=100)

    if st.button("Submit"):
        collected_data = {
            "Data": data_input,
            "Countries": countries_input,
            "Local": local_input,
            "Industry Sector": industry_sector_input,
            "Potential Accident Level": potential_accident_level_input,
            "Genre": genre_input,
            "Employee or Third Party": employee_third_party_input,
            "Critical Risk": critical_risk_input,
            "Description": description_input
        }
        response = process_user_input(collected_data, nlp_model)
        st.text_area("Chatbot Response:", value=response, height=100, max_chars=500, key="chat_response", disabled=True)


def process_user_input(user_input, model):
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    extractGlove(url)
    df = pd.DataFrame([user_input])
    df['Date'] = pd.to_datetime(df['Data']).dt.date
    df.rename(columns={'Countries':'Country', 'Genre':'Gender','Employee or Third Party' :'Employee Type'}, inplace=True)
    df['Employee Type'] = df['Employee Type'].replace(['Third Party (Remote)'], 'Third Party')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.apply(lambda x : x.year)
    df['Month'] = df.Date.apply(lambda x : x.month)
    df['Day'] = df.Date.apply(lambda x : x.day)
    df['Weekday'] = df.Date.apply(lambda x : x.day_name())
    df['WeekofYear'] = df.Date.apply(lambda x : x.weekofyear)
    df['Season'] = df['Month'].apply(monthToseasons)
    stop_words = set(stopwords.words('english'))
    df['Cleaned_Description'] = df['Description'].apply(lambda x : x.lower())
    df['Cleaned_Description'] = df['Cleaned_Description'].apply(lambda x : replace_words(x))
    df['Cleaned_Description'] = df['Cleaned_Description'].apply(lambda x: remove_punctuation(x))
    df['Cleaned_Description'] = df['Cleaned_Description'].apply(lambda x: lemmatize(x))
    df['Cleaned_Description'] = df['Cleaned_Description'].apply(lambda x: re.sub(' +', ' ', x))
    df['Cleaned_Description'] = df['Cleaned_Description'].apply(lambda x: remove_stopwords(x))
    df['line_length'] = df['Cleaned_Description'].str.len()
    df['nb_words'] = df['Cleaned_Description'].apply(lambda x: len(x.split(' ')))
    df['Employee type'] = df['Employee Type'].str.replace(' ', '_')
    df['Critical Risk'] = df['Critical Risk'].str.replace('\n', '').str.replace(' ', '_')
    embeddings_index = embedded_words()
    ind_featenc_df = pd.DataFrame()
    df['Season'] = df['Season'].replace('Summer', 'aSummer').replace('Autumn', 'bAutumn').replace('Winter', 'cWinter').replace('Spring', 'dSpring')
    ind_featenc_df['Season'] = LabelEncoder().fit_transform(df['Season']).astype(np.int8)
    df['Weekday'] = df['Weekday'].replace('Monday', 'aMonday').replace('Tuesday', 'bTuesday').replace('Wednesday', 'cWednesday').replace('Thursday', 'dThursday').replace('Friday', 'eFriday').replace('Saturday', 'fSaturday').replace('Sunday', 'gSunday')
    ind_featenc_df['Weekday'] = LabelEncoder().fit_transform(df['Weekday']).astype(np.int8)
    ind_featenc_df['Potential Accident Level'] = LabelEncoder().fit_transform(df['Potential Accident Level']).astype(np.int8)
    Country_dummies = pd.get_dummies(df['Country'], columns=["Country"], drop_first=True)
    Local_dummies = pd.get_dummies(df['Local'], columns=["Local"], drop_first=True)
    Gender_dummies = pd.get_dummies(df['Gender'], columns=["Gender"], drop_first=True)
    IS_dummies = pd.get_dummies(df['Industry Sector'], columns=['Industry Sector'], prefix='IS', drop_first=True)
    EmpType_dummies = pd.get_dummies(df['Employee type'], columns=['Employee type'], prefix='EmpType', drop_first=True)
    CR_dummies = pd.get_dummies(df['Critical Risk'], columns=['Critical Risk'], prefix='CR', drop_first=True)
    ind_featenc_df = ind_featenc_df.join(Country_dummies.reset_index(drop=True)).join(Local_dummies.reset_index(drop=True)).join(Gender_dummies.reset_index(drop=True)).join(IS_dummies.reset_index(drop=True)).join(EmpType_dummies.reset_index(drop=True)).join(CR_dummies.reset_index(drop=True))
    ind_featenc_df = df[['Year','Month','Day','WeekofYear']].reset_index(drop=True).join(ind_featenc_df.reset_index(drop=True))
    ind_glove_df = [sent2vec(x) for x in tqdm(df['Cleaned_Description'])]
    ind_tfidf_df = pd.DataFrame()
    for i in range(1, 4):
     vec_tfidf = TfidfVectorizer(max_features=10, norm='l2', stop_words='english', lowercase=True, use_idf=True, ngram_range=(i, i))
     X = vec_tfidf.fit_transform(df['Cleaned_Description']).toarray()
     tfs = pd.DataFrame(X, columns=["TFIDF_" + n for n in vec_tfidf.get_feature_names_out()])
     ind_tfidf_df = pd.concat([ind_tfidf_df.reset_index(drop=True), tfs.reset_index(drop=True)], axis=1)
    ind_feat_df = ind_featenc_df.join(pd.DataFrame(ind_glove_df).iloc[:,0:30].reset_index(drop=True))
    scaler_X = StandardScaler()
    ind_tfidf_df.iloc[:,:6] = scaler_X.fit_transform(ind_tfidf_df.iloc[:,:6])
    text_samples = 'observing pulp overflow overflow reception drawer thickener filter operator approach verify operation pump making sure stopped press keypad start pump getting start proceeds remove guard manipulates motor pump transmission strip left hand imprisoned pulley motor transmission belt'  # Replace with your actual text samples
    #categorical_samples =ind_feat_df.values[:len(text_samples)]
    result = predict_text_and_categorical(text_samples, ind_tfidf_df,model)
    #model.predict([ind_tfidf_df,ind_tfidf_df])
    #return f"Received input: [0.7049883  0.10264347 0.08182887 0.0828189  0.02772051]"
    return f"Received input: {result[0]}"

def monthToseasons(x):
    if x in [9, 10, 11]:
        season = 'Spring'
    elif x in [12, 1, 2]:
        season = 'Summer'
    elif x in [3, 4, 5]:
        season = 'Autumn'
    elif x in [6, 7, 8]:
        season = 'Winter'
    return season
def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
def replace_words(text):
    # Replace 'word_to_replace' with 'replacement_word'
    text = text.replace('word_to_replace', 'replacement_word')
    return text
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    # Your implementation to lemmatize words goes here
    # For example:
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
def remove_punctuation(text):
    # Remove punctuation from the text
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)
def sent2vec(s):
    stop_words = stopwords.words('english')
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())
    
def des_cleaning(text):

    # Initialize the object for Lemmatizer class
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Set the stopwords to English
    stopwords = nltk.corpus.stopwords.words('english')

    # Normalize the text in order deal with accented words and unicodes
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())

    # Consider only alphabets and numbers from the text
    words = re.sub(r'[^a-zA-Z.,!?/:;\"\'\s]', '', text).split()

    # Consider the words which are not in stopwords of english and lemmatize them
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lems = [lemmatizer.lemmatize(i) for i in words if i not in stopwords]

    # #remove non-alphabetical characters like '(', '.' or '!'
    # alphas = [i for i in lems if (i.isalpha() or i.isnumeric()) and (i not in stopwords)]

    words = [w for w in lems if len(w)>2]

    return words
    
def predict_text_and_categorical(text_samples, categorical_samples, model):
    # Preprocess text data
    text_data = [" ".join(des_cleaning(text)) for text in text_samples]
    tokenizer = Tokenizer(num_words=10000)
    x = tokenizer.texts_to_sequences(text_data)
    input_1_data = pad_sequences(x, maxlen=100)
    
    # Preprocess categorical data
    input_2_data = np.array(categorical_samples)
    original_data = categorical_samples.iloc[0].values  # Extract the row as a NumPy array
    length = len(text_samples)
    replicated_data = np.tile(original_data, (length, 1))
    expanded_data = np.zeros((length, 85))
    expanded_data[:, :30] = replicated_data
    expanded_df = pd.DataFrame(expanded_data)
    predictions = model.predict([input_1_data,expanded_df])
    print(predictions[0])
    return predictions


if __name__ == "__main__":
    main()
