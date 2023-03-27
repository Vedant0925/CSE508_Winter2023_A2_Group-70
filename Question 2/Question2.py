import pandas as pd 
import numpy as np 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 


df = pd.read_csv('Question 2/BBC News Train.csv') 
df = df[['Text', 'Category']] 

nltk.download('stopwords') 
stop_words = stopwords.words('english') 
porter = PorterStemmer() 
#pre processing dataset
def pre_process(text): 
    text = text.lower() 
    text = re.sub('[^a-zA-Z]', ' ', text) 
    words = text.split() 
    words = [porter.stem(word) for word in words if word not in stop_words] 
    return ' '.join(words) 

df['Text'] = df['Text'].apply(pre_process) 

#convert the text into TF-IDF vectors with TF-ICF weighting scheme 
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=5000) 
X = vectorizer.fit_transform(df['Text']) 

#split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, df['Category'], test_size=0.3, stratify=df['Category'], random_state=42) 

#train the Naive Bayes classifier with TF-ICF 
clf = MultinomialNB() 
clf.fit(X_train, y_train) 

#predict on the test set 
y_pred = clf.predict(X_test) 

#evaluate the performance of the classifier 
print('Accuracy:', accuracy_score(y_test, y_pred)) 
print('Precision:', precision_score(y_test, y_pred, average='macro', zero_division=0)) 
print('Recall:', recall_score(y_test, y_pred, average='macro')) 
print('F1 score:', f1_score(y_test, y_pred, average='macro'))