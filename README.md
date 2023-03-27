# CSE508_Winter2023_A2_Group-70

Ans 1)
First we did the pre processing 
(i) Relevant Text Extraction 
1st Printed the initial 5 files before and preprocessing.
We will also unpack the compressed file to get the folder of our dataset to use it. 
Then loaded all the files one by one using BeautifulSoup.Then we separated the title and the text part of the files and concatenated only the title and text part of the files. Then we again wrote this concatenated string back to the original files and then printed the first 5 files.

(ii) Preprocessing
We loaded the dataset obtained after Title and Text extraction, Using the nltk library, we 1st lowercase the text.
 Then we did tokenization, in which we also considered blank spaces as individual tokens by replacing them with ‘@’ and after tokenization replacing them back to blank space, so they don’t get deleted during tokenization.
We also considered a-b-c as three different tokens ‘a’ , ‘b’ , ‘c’. 
Then we removed stop words from the tokens using nltk library.
Then we removed punctuations using isalpha() function
Then we removed blank spaces by removing ‘ ‘ from tokens.
We then saved our token in the respective original files as string of list of tokens.

Then we created a vocab matrix having all the unique words. They we created functions for each of the different Weighting schemes using their given formula. Then we used these functions to create different tf-idf matrices. Then we created a query vector using an example query. Then we used this query vector to calculate the tf-idf scores of tf-idf matrices of different weighing schemes.

The Pros and cons of various schemes are 

Even though it is very simple, a binary weighting scheme is suitable when we only want to know whether a term occurs in a document or not and not how many times it occurs.

The raw count weighting scheme is simple to implement and indicates the term's importance in the document. However, it can be biased towards longer documents, and will work well in case of longer documents.

The term frequency (TF) weighting scheme is widely used in practice as it considers that longer documents will have higher term frequencies. However, it is sensitive to noise and can give high weightage to useless terms that appear again and again.

The log normalisation scheme helps to reduce the effect of longer documents by taking the logarithm of the term frequency. It can also reduce the impact of terms that frequently occur across all documents, which are not very informative. But it does not take in account of the document frequency.

A double normalisation scheme further reduces the impact of longer documents by dividing the term frequency by the maximum term frequency of any term in the document. It is useful when we want to emphasise the most critical terms in the document.But it does not take in account of the document frequency.

Jaccard Coefficient
This function determines the Jaccard coefficient between the tokens in the query and each document using a list of documents represented as sentences and a query string. The Jaccard coefficient, which is calculated by dividing the size of two sets, intersection by their union, calculates how similar they are. The top 10 most comparable docs are then returned after the code ranksthe documents according to their Jaccard coefficient.



Question 2:
This was the Accuracy and precision before using N-grams to improve the classifier. 
Accuracy: 0.22550335570469798
Precision: 0.045100671140939595
Recall: 0.2
F1 score: 0.07360350492880614

After implementing the following changes
- Added an ngram_range of (1,3) and max_features of 5000 to the TfidfVectorizer. This can help capture more contextual information and higher-frequency terms.
- Used the MultinomialNB classifier, which is well-suited for text classification tasks.
- Removed the self-implemented NBClassifier, and instead used the scikit-learn implementation for better performance.

This was updated results with 70-30 data Training testing ratio

Accuracy: 0.9776286353467561
Precision: 0.9783150183150184
Recall: 0.9761521743168687
F1 score: 0.9770407020164301

If I change the training testing to 50-50 then these are updated results 
Accuracy: 0.9651006711409396
Precision: 0.9661964614333035
Recall: 0.9630762546455977
F1 score: 0.9642896798412298

Hence the final most optimum accuracy that is achieved with 70-30 training/ testing ration is.

Accuracy: 0.9776286353467561
Precision: 0.9783150183150184
Recall: 0.9761521743168687
F1 score: 0.9770407020164301


How to run code: 
There is a question 2 folder along with dataset, just run the python file in terminal. 

Question 3:
Feed input data into a pandas data frame and filter to only consider rows with ‘qid:4’.
Calculate DCG(max possible) using DCG = sum((2 ^ relevance scores - 1) / log2(rank + 1)) for given relevance scores. Sort data based on relevance scores and save it to a new file(qid_4_sorted_pairs.txt here)
There might be multiple arrangements of items with the same relevance score; they will not impact the maximum DCG as long as the items maintain their relative order based on the descending relevance score. As long as the items with a higher relevance score appear before those with a lower relevance score, any permutations among items with the same relevance score will still result in the same maximum DCG value. Therefore, the number of files that can be created while maintaining the maximum DCG should be one.
Calculate ndcg using custom function ndcg that takes arguments max dcg, relevance scores, and k to calculate top k relevance scores. If no k is provided, ndcg is calculated for the entire dataset. K is defined as 50 here.
Iterate through threshold values for feature 75 and calculate True Positives (TP), False Positives (FP), and False Negatives (FN).
Then compute precision and recall for each threshold and plot curve.

ndcg at position 50:  0.35612494416255847
ndcg for the entire dataset:  0.5784691984582588
