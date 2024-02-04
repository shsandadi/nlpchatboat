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
    nlp_model = load_model()
    tf_version = tf.__version__
    st.sidebar.header(f"Information (TensorFlow Version: {tf_version})")
    st.sidebar.markdown("This is a simple NLP chatbox for handling accident-related queries.")
    user_input = st.text_input("User Input:", "")
    st.text_area("Chatbot Response:", height=100, max_chars=500, value="", key="chat_response", disabled=True)
    if st.button("Submit"):
        #response = process_user_input(user_input, nlp_model)
        #nlp_model='abc'
        response = process_user_input(user_input, nlp_model)
        st.text_area("Chatbot Response:", value=response, key="chat_response", disabled=True, background="#F0F8FF", font_color="#333", font_family="Arial, sans-serif")

def process_user_input(user_input, model):
    return "Response from your model here."

if __name__ == "__main__":
    main()
