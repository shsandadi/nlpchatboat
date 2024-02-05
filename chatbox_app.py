import pickle
import streamlit as st
import tensorflow as tf
import pandas as pd

def load_model():
    model_path = r'nlp_chatbot.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    st.set_page_config(page_title="NLP Chatbox for Accident", page_icon="🚑", layout="centered")
    st.title("NLP Chatbox for Accident")
    st.markdown('<style>h1{color: #336699;}</style>', unsafe_allow_html=True)
    nlp_model = load_model()
    tf_version = tf.__version__
    st.sidebar.header(f"This is chatbot of Group 6 to predict the accident level")
    st.sidebar.markdown("This is a simple NLP chatbox for handling accident-related queries.")

    # User inputs for each column
    data_input = st.text_input("Data:")
    countries_input = st.text_input("Countries:")
    local_input = st.text_input("Local:")
    industry_sector_input = st.text_input("Industry Sector:")
    potential_accident_level_input = st.text_input("Potential Accident Level:")
    genre_input = st.text_input("Genre:")
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
    # Process the collected data and return a response from your model here.
    # For now, we'll just return a string representation of the user_input dictionary.
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
    return f"Received input: {df.shape}"

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

if __name__ == "__main__":
    main()
