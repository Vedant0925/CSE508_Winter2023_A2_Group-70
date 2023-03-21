import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = "/Users/vedant/Downloads/IR-assignment-2-data (2).txt"  
data = pd.read_csv(file_path, sep=" ", header=None)

data_qid4 = data[data[1] == 'qid:4']
relevance_scores = data_qid4[0].values
sorted_scores = np.sort(relevance_scores)[::-1]

max_dcg = np.sum((2 ** sorted_scores - 1) / np.log2(np.arange(2, len(sorted_scores) + 2)))


data_qid4_sorted = data_qid4.iloc[np.argsort(-relevance_scores)]
data_qid4_sorted.to_csv("qid_4_sorted_pairs.txt", sep=" ", header=None, index=None)
def ndcg(relevance_scores, max_dcg, k=None):
    if k is not None:
        relevance_scores = relevance_scores[:k]
    dcg = np.sum((2 ** relevance_scores - 1) / np.log2(np.arange(2, len(relevance_scores) + 2)))
    return dcg / max_dcg


ndcg_50 = ndcg(relevance_scores, max_dcg, k=50)
print("ndcg at position 50: ",ndcg_50)


ndcg_all = ndcg(relevance_scores, max_dcg)
print("ndcg for entire dataset: ",ndcg_all)


feature_75_values = data_qid4[75].apply(lambda x: float(x.split(":")[1])).values
relevant_labels = (relevance_scores > 0).astype(int)


precision, recall = [], []
for threshold in np.linspace(0, max(feature_75_values), num=100):
    predictions = (feature_75_values > threshold).astype(int)
    tp = np.sum(predictions * relevant_labels)
    fp = np.sum(predictions * (1 - relevant_labels))
    fn = np.sum((1 - predictions) * relevant_labels)

    precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0

    precision.append(precision_value)
    recall.append(recall_value)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for qid:4 using Feature 75")
plt.show()
