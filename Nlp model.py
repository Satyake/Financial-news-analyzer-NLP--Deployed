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
from tensorflow.keras.layers import Embedding,LeakyReLU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data=pd.read_csv('all-data.csv',encoding='ISO-8859-1')
LE=LabelEncoder()
y=data.iloc[:,[0]].values
x=data.iloc[:,[1]].values
y=LE.fit_transform(y)
x=x.tolist()
LEMMATIZER=WordNetLemmatizer()
corpus=[]
for i in range(0,len(x)):
    lines=re.sub('[^a-zA-Z]',' ',str(x[i]))
    #lines=line.sub('https?:\/\/.*[\r\n]*','',lines)
    lines=lines.lower()
    lines=lines.split()
    lines=[LEMMATIZER.lemmatize(j) for j in lines if j not in stopwords.words('english')]
    lines=' '.join(lines)
    corpus.append(lines)
len(corpus)
vectors=Word2Vec(corpus)
sentence_length=80
OHR=[one_hot(k,4000) for k in corpus]
embeddings=pad_sequences(OHR,sentence_length)
GNB=GaussianNB()

x_train,x_test,y_train,y_test=train_test_split(embeddings,y,train_size=0.7,shuffle=True)
GNB.fit(x_train,y_train)
predicted=GNB.predict(x_test)
RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
predicted_rf=RFC.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(predicted_rf,y_test)

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional
model=Sequential()
model.add(Embedding(4000,30,input_length=sentence_length))
model.add(Bidirectional(LSTM(300,activation='relu',return_sequences=True)))
model.add(LeakyReLU(alpha=0.3))
model.add(LSTM(200,activation='relu',return_sequences=False))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=40,batch_size=15)


model.save('NLPTrained.h5')
corpus1=[]
preds=model.predict(x_test)
preds= np.argmax(preds,axis=-1)
accuracy_score(preds,y_test)
confusion_matrix(preds,y_test)

unknown='Alas '
unknown=np.array(unknown)
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
np.argmax(predicted,axis=-1)














