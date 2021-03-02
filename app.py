from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd 
import numpy as np 
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot





# Flask utils
from flask import Flask, redirect, url_for, request, render_template
app=Flask(__name__)

model=load_model('NLPTrained.h5')
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/pred',methods=['POST'])
def pred():
    x=[text for text in request.form.values()]
    LEMMATIZER=WordNetLemmatizer()
    unknown=np.array(x)
    corpus1=[]
    sentence_length=80

    lines1=re.sub('[^a-zA-Z]',' ',str(unknown))
      #lines=line.sub('https?:\/\/.*[\r\n]*','',lines)
    lines1=lines1.lower()
    lines1=lines1.split()
    lines1=[LEMMATIZER.lemmatize(j) for j in lines1 if j not in stopwords.words('english')]
    lines1=' '.join(lines1)
    corpus1.append(lines1)
    OHR=[one_hot(k,4000) for k in corpus1]
    embeddings=pad_sequences(OHR,sentence_length)
    embeddings=embeddings.reshape(-1,1)
    embeddings=np.transpose(embeddings)
    predicted=model.predict(embeddings)
    #predicted=np.argmax(predicted,axis=-1)

    return render_template('index.html', prediction_text='Mood = {} Softmax Distr between Negative|Neutral|Positive'.format(predicted))

if __name__== '__main__':
    app.run(host='0.0.0.0',port=8080)
