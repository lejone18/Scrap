import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import spacy
import streamlit as st
from joblib import load

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Define the preprocess function
def preprocess(bullying_words):
    if pd.notnull(bullying_words):  
        doc = nlp(bullying_words)
        filtered_tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.like_num:
                filtered_tokens.append(token.lemma_)
        return ' '.join(filtered_tokens)
    else:
        return ''  

# Load the trained model
pipeline = load('lejone99.joblib')

# Streamlit app
def main():
    st.title('Bullying Detection App')
    text = st.text_input('Enter text to check for bullying:')
    if st.button('Check'):
        processed_text = preprocess(text)
        prediction = pipeline.predict([processed_text])
        st.write(f'The text is classified as: {prediction[0]}')

if __name__ == '__main__':
    main()
