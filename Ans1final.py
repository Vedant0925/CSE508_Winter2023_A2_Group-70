from bs4 import BeautifulSoup
import os

import nltk
nltk.download('punkt')


import nltk
nltk.download('punkt')
nltk.download('stopwords')
import itertools




from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math
from math import log
import numpy as np




sentences =[]

def preprocessing():
    

    print("Printing 5 sample before any changes \n")

    for i in range (1,6):
        url = "CSE508_Winter2023_Dataset\cranfield000"+str(i)
        with open(url) as infile:
            soup = BeautifulSoup(infile) 
        print(soup.prettify())
        print("\n")

    ########### Printing 5 sample before



    for i in range (1,1401):
        if (i <10):
            url = "CSE508_Winter2023_Dataset\cranfield000"+str(i)
        elif (i<100):
            url = "CSE508_Winter2023_Dataset\cranfield00"+str(i)
        elif (i<1000):
            url = "CSE508_Winter2023_Dataset\cranfield0"+str(i)
        elif (i<10000):
            url = "CSE508_Winter2023_Dataset\cranfield"+str(i)
        
        
        with open(url) as infile:
            soup = BeautifulSoup(infile) 
        
        el = soup.find("text") 

        el_text = el.string 
        if (i <10):
            url2 = "CSE508_Winter2023_Dataset\cranfield000"+str(i)
        elif (i<100):
            url2= "CSE508_Winter2023_Dataset\cranfield00"+str(i)
        elif (i<1000):
            url2 = "CSE508_Winter2023_Dataset\cranfield0"+str(i)
        elif (i<10000):
            url2 = "CSE508_Winter2023_Dataset\cranfield"+str(i)
        
        f = open(url2, "w")
        f.write((soup.title.string))
        f.close
        f = open(url2, "a")
        f.write(el_text)




    print("Printing 5 samples after the changes \n")


    for i in range (1,6):
        f = open("CSE508_Winter2023_Dataset\cranfield000"+str(i), "r")
        print(f.read())
        print("\n")

    ########### Printing 5 sample after



    import nltk
    nltk.download('punkt')


    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    import itertools




    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords



    for i in range (1,1401):
        tokens =[]
        new_words =[]
        filtered_tokens=[]
        final_words=[]

        
        if (i <10):
            url2 = "CSE508_Winter2023_Dataset\cranfield000"+str(i)
        elif (i<100):
            url2= "CSE508_Winter2023_Dataset\cranfield00"+str(i)
        elif (i<1000):
            url2 = "CSE508_Winter2023_Dataset\cranfield0"+str(i)
        elif (i<10000):
            url2 = "CSE508_Winter2023_Dataset\cranfield"+str(i)


        ### Tokenizing 
        f = open(url2, "r")
        text = f.read()

        lower_text = text.lower()
        text = lower_text
        if (i<6):
            print("case-"+str(i)+" text after lowercasing the text")
            print(text)
            print("\n")

        # tokens = word_tokenize(text)
        text = text.replace(' ', '@')
        text = text.replace('-', ' ')
        
        # ll = [[' ',word_tokenize(w)] for w in text.split()]
        # tokens = list(itertools.chain(*list(itertools.chain(*ll))))
        # tokens = text.split(' ')
        tokens = word_tokenize(text)
        tokens = list(map(lambda x: x.replace('@', ' '), tokens))
        
        if (i<6):
            print("case-"+str(i)+"  Tokens are")
            print(tokens)
            print("\n")



        ## Removing Stop Words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
        if (i<6):
            print("case-"+str(i)+"  Tokens after removing stop Words")
            print(filtered_tokens) 
            print("\n")


        ### Remove Punctuations
        new_words=[word.lower() for word in filtered_tokens if (word.isalpha() or word==' ')]
        if (i<6):
            print("case-"+str(i)+"  Tokens after removing punctiations")
            print(new_words)
            print("\n")


        ### Remove Blank Spaces 
        final_words=[word.lower() for word in new_words if word.isalpha() ]
        final_text=""
        for j in range(0,len(final_words)):
            final_text = final_text + final_words[j]+" "
        
        
        if (i<6):
            print("case-"+str(i)+" Tokens after removing Blank Spaces")
            print(final_words)
            print("\n")

        f = open(url2, "w")
        f.write(final_text)
        f.close()   


preprocessing()

    


sentences =[]

for i in range (1,1401):
        
        if (i <10):
            url2 = "CSE508_Winter2023_Dataset/cranfield000"+str(i)
        elif (i<100):
            url2= "CSE508_Winter2023_Dataset/cranfield00"+str(i)
        elif (i<1000):
            url2 = "CSE508_Winter2023_Dataset/cranfield0"+str(i)
        elif (i<10000):
            url2 = "CSE508_Winter2023_Dataset/cranfield"+str(i)
          

        f = open(url2, "r")
        text = f.read()
        final_words = word_tokenize(text)
        final_text=""
        for j in range(0,len(final_words)):
            final_text = final_text + final_words[j]+" "
        sentences.append(final_text)



tokens = [sentence.split() for sentence in sentences]  ## tokens for all the different words
vocab = set()       ## creating a vocabulary set (all the uniwue words)

for sentence in tokens:
    for word in sentence:
        vocab.add(word)
        # print(word)
vocab = sorted(list(vocab))
vocab_matrix = vocab


def binary_tf(term, document):
    return int(term in document) ## Checks if term is present in the doc
def raw_tf(term, document):
    return document.count(term)  ## returns the number of occurences of the term
def tf(term, document):
    return document.count(term) / sum(document.count(t) for t in set(document.split()))  ## returns the value of term freuquency using the formula (Number of times term t appears in a document) / (Total number of terms in the document)

def log_tf(term, document):
    return math.log(1 + document.count(term)) ## Returns the log of (1+frequency)
def double_norm_tf(term, document):
    tf = document.count(term)
    max_tf = max(document.count(t) for t in set(document.split())) ## returns the normalized value of term frequency
    return 0.5 + 0.5 * (tf / max_tf)


def tfidf(documents, vocab_matrix, tf_func):
    # Calculate document frequency for each term in the vocabulary
    doc_freq = {}
    for term in vocab_matrix:
        doc_freq[term] = sum(1 for doc in documents if term in doc)

    # Calculate TF-IDF matrix for each document
    tfidf_matrix = []
    for doc in documents:
        tfidf_vector = []
        for term in vocab_matrix:
            tf = tf_func(term, doc)
            idf = math.log(len(documents) / (doc_freq[term] ))
            tfidf_vector.append(tf * idf)
        tfidf_matrix.append(tfidf_vector)

    return tfidf_matrix


binary_tfidf_matrix = tfidf(sentences, vocab_matrix, binary_tf)
raw_tfidf_matrix = tfidf(sentences, vocab_matrix, raw_tf)
tf_tfidf_matrix = tfidf(sentences, vocab_matrix, tf)
log_tfidf_matrix = tfidf(sentences, vocab_matrix, log_tf)
double_norm_tfidf_matrix = tfidf(sentences, vocab_matrix, double_norm_tf)

def construct_query_vector(query, vocab_matrix):
    query_terms = query.split()
    query_vector = [0] * len(vocab_matrix)
    for i, term in enumerate(vocab_matrix):
        if term in query_terms:
            
            query_vector[i] = 1
    return query_vector

def compute_tfidf_score(query_vector, tfidf_matrix):
    scores = []
    for i, tfidf_vector in enumerate(tfidf_matrix):
        score = sum(q * d for q, d in zip(query_vector, tfidf_vector))
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# Example usage

vocab_matrix = set([term for doc in sentences for term in doc.split()])

binary_tfidf_matrix = tfidf(sentences, vocab_matrix, binary_tf)

query = "experimental investigation"
query_vector = construct_query_vector(query, vocab_matrix)

query_scores = compute_tfidf_score(query_vector, binary_tfidf_matrix)


print("Query:", query)
print("Top 5 relevant documents for Binary:")
for i, score in query_scores[:5]:
    print(f"Document {i+1}: {sentences[i]} \n (score: {score:.2f})\n")


query_scores = compute_tfidf_score(query_vector, raw_tfidf_matrix)


print("Query:", query)
print("Top 5 relevant documents for Raw Method:")
for i, score in query_scores[:5]:
    print(f"Document {i+1}: {sentences[i]} \n(score: {score:.2f})\n")


query_scores = compute_tfidf_score(query_vector, tf_tfidf_matrix)


print("Query:", query)
print("Top 5 relevant documents for Term Frequency:")
for i, score in query_scores[:5]:
    print(f"Document {i+1}: {sentences[i]} \n(score: {score:.2f})\n")


query_scores = compute_tfidf_score(query_vector, log_tfidf_matrix)


print("Query:", query)
print("Top 5 relevant documents for log normalization:")
for i, score in query_scores[:5]:
    print(f"Document {i+1}: {sentences[i]} \n (score: {score:.2f})\n ")


query_scores = compute_tfidf_score(query_vector, double_norm_tfidf_matrix)


print("Query:", query)
print("Top 5 relevant documents for Double Normalization:")
for i, score in query_scores[:5]:
    print(f"Document {i+1}: {sentences[i]} \n (score: {score:.2f}) \n")



# Sample documents and query
# documents = sentences
query = "experimental investigation"

# Convert documents and query to sets of tokens
doc_sets = [set(doc.split()) for doc in sentences]
query_set = set(query.split())



# Computing Jaccard coefficient for each document for the given query
jaccard_scores = []
for i, doc_set in enumerate(doc_sets):
    intersection = len(doc_set.intersection(query_set))
    union = len(doc_set.union(query_set))
    
    score = intersection / union
    jaccard_scores.append((i, score))

# Rank documents by Jaccard coefficient
ranked_docs = sorted(jaccard_scores, key=lambda x: x[1], reverse=True)[:10]

# Print top 10 documents and their scores
for i, score in ranked_docs:
    print(f"Document {i+1}: Jaccard Coefficient = {score:.4f}")

