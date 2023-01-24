import streamlit as st
import nltk
import pickle
import string
import bz2file as bz2
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
stoplist = set(stopwords.words('english'))
nltk.download('punkt')

ps = PorterStemmer()
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

model=decompress_pickle('model.pbz2')
vectorizer=decompress_pickle('vectorizer.pbz2')

#model=pickle.load(open('model.pkl', 'rb'))
#vectorizer=pickle.load(open('vectorizer.pkl', 'rb'))

#vectorizer=pickle.load(open('vectorizer.pkl','rb'))

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[token for token in text if token.isalnum()==True]
    text=[token for token in text if token not in stoplist and token not in string.punctuation]
    text=[ps.stem(token) for token in text]
    return " ".join(text)

def checker(txt):
    txt=transform_text(txt)
    X = vectorizer.transform([txt])
    print(X)
    ans=model.predict(X)
    print(ans)
    if ans[0]==0:
        return "NotSpam"
    else:
        return "Spam"




st.title('SMS Checker')

with st.form("my_form"):
   st.write("Inside the form")
   txt = st.text_area('Text to analyze', )


   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       print(txt)
       st.write("slider",checker(txt))

st.write("Outside the form")

