import pandas as pd
import pickle as pk
import streamlit as st
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import string
import re
# Loading both the LR model and vectorizer
model= pk.load(open("model.pkl", "rb"))
scaler= pk.load(open("scaler.pkl", "rb"))


st.header("Automated Sentiment Analysis for Movie Reviews")

review=st.text_area("Enter movie review:",height=150,max_chars=1000)

def clean_text(text):
        text=" ".join(word for word in text.split() if word.lower() not in stopwords.words("english"))
        
        text=text.lower()
        
        text=re.sub(r"https\S+|www\S+http\S+","",text,flags=re.MULTILINE)
        text=re.sub(r"@[\w-]+","",text)
        text=re.sub(r"\d+","",text)
        text=text.translate(str.maketrans("","",string.punctuation))

        return text
        
       
def sentiment_check(text):
        
        cleaned_text=clean_text(text)
        
        if not cleaned_text:
            return None
        else:
            text_vector=scaler.transform([cleaned_text]).toarray()
            result=model.predict(text_vector)
            return result

        
def validate_input(text):
        if not text:
                return "Review cannot be empty."
        if len(text) < 10:
                return "Review must be at least 10 characters long."

        
if st.button("predict"):
        val_message=validate_input(review)
        if val_message:
            st.error(val_message)
        else:
            result=sentiment_check(review)
            if result is None:
                   st.error("Review cannot be empty/containing special characters/links/numbers")
            elif result==1:
                    st.write("Positive Review")
            elif result==0:
                    st.write("Negative Review")
