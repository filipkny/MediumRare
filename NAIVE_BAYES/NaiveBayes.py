from collections import defaultdict
import numpy as np

# noinspection SpellCheckingInspection
class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = defaultdict(int)
        self.logprior = {}
        self.bigdoc = defaultdict(list)
        self.loglikelihoods = defaultdict(defaultdict)
        self.V = []
        self.n = n_gram

    def compute_prior_and_bigdoc(self, training_set, training_labels):
        '''
        Computes the prior and the bigdoc (from the book's algorithm)
        :param training_set:
            a list of all documents of the training set
        :param training_labels:
            a list of labels corresponding to the documents in the training set
        :return:
            None
        '''
        for x, y in zip(training_set, training_labels):
            all_words = x.split(" ")
            if self.n == 1:
                grams = all_words
            else:
                grams = self.words_to_grams(all_words)

            self.prior[y] += len(grams)
            self.bigdoc[y].append(x)

    def compute_vocabulary(self, documents):
        vocabulary = set()

        for doc in documents:
            for word in doc.split(" "):
                vocabulary.add(word.lower())

        return vocabulary

    def count_word_in_classes(self):
        counts = {}
        for c in list(self.bigdoc.keys()):
            docs = self.bigdoc[c]
            counts[c] = defaultdict(int)
            for doc in docs:
                words = doc.split(" ")
                for word in words:
                    counts[c][word] += 1

        return counts

    def train(self, training_set, training_labels, alpha=1):
        # Get number of documents
        N_doc = len(training_set)

        # Get vocabulary used in training set
        self.V = self.compute_vocabulary(training_set)

        # Create bigdoc
        for x, y in zip(training_set, training_labels):
            self.bigdoc[y].append(x)

        # Get set of all classes
        all_classes = set(training_labels)

        # Compute a dictionary with all word counts for each class
        self.word_count = self.count_word_in_classes()

        # For each class
        for c in all_classes:
            # Get number of documents for that class
            N_c = float(sum(training_labels == c))

            # Compute logprior for class
            self.logprior[c] = np.log(N_c / N_doc)

            # Calculate the sum of counts of words in current class
            total_count = 0
            for word in self.V:
                total_count += self.word_count[c][word]

            # For every word, get the count and compute the log-likelihood for this class
            for word in self.V:
                count = self.word_count[c][word]
                self.loglikelihoods[c][word] = np.log((count + alpha) / (total_count + alpha * len(self.V)))

    def predict(self, test_doc):
        sums = {
            0: 0,
            1: 0,
        }
        for c in self.bigdoc.keys():
            sums[c] = self.logprior[c]
            words = test_doc.split(" ")
            for word in words:
               if word in self.V:
                   sums[c] += self.loglikelihoods[c][word]

        return sums


# doc1 = "just plain boring"                      # -
# doc2 = "entirely predictable and lacks energy"  # -
# doc3 = "no surprises and very few laughs"       # -
# doc4 = "very powerful"                          # +
# doc5 = "the most fun film of the summer"        # +
#
# training_set = [doc1, doc2, doc3, doc4, doc5]
# training_labels = np.array([0, 0, 0, 1 ,1])
#
# doc6 = "predictable with no fun" # ?
#
# NBclassifier = NaiveBayesClassifier(n_gram=1)
# NBclassifier.train(training_set,training_labels)
#
# result = NBclassifier.predict(doc6)
# print(np.exp(result))

# Big file stuff
import string
import json

with open("reviews.json", mode="r", encoding="utf-8") as f:
  reviews = json.load(f)

sentiment_numerical_val = {
    'NEG': 0,
    'POS': 1
}

import pprint

def split_review_data(reviews, split=900, remove_punc=False, separation=" "):
    training_set = []
    training_labels = []
    validation_set = []
    validation_labels = []

    for i, r in enumerate(reviews):
        if i==0: print(str(r['content'])); print(dict(r).keys())
        cv = int(r["cv"])
        sent = sentiment_numerical_val[r["sentiment"]]
        content_string = ""
        for sentence in r["content"]:
            for word in sentence:
                content_string += word[0].lower() + separation

        if remove_punc:
            exclude = set(string.punctuation)
            content_string = ''.join(character for character in content_string if character not in exclude)

        if 0 < cv < split:
            training_set.append(content_string)
            training_labels.append(sent)
        else:
            validation_set.append(content_string)
            validation_labels.append(sent)

    return training_set, np.array(training_labels), validation_set, np.array(validation_labels)

def evaluate_predictions(validation_set,validation_labels,trained_classifier):
  correct_predictions = 0
  predictions_list = []
  prediction = -1

  for dataset,label in zip(validation_set, validation_labels):
    probabilities = trained_classifier.predict(dataset)
    if probabilities[0] >= probabilities[1]:
      prediction = 0
    elif  probabilities[0] < probabilities[1]:
      prediction = 1

    if prediction == label:
      correct_predictions += 1
      predictions_list.append("+")
    else:
      predictions_list.append("-")

  print("Predicted correctly {} out of {} ({}%)".format(correct_predictions,len(validation_labels),round(correct_predictions/len(validation_labels)*100,5)))
  return predictions_list, round(correct_predictions/len(validation_labels)*100)


training_set, training_labels, validation_set, validation_labels = split_review_data(reviews)

import time

start = time.time()

NBclassifier = NaiveBayesClassifier()
NBclassifier.train(training_set, training_labels, alpha=1)
results, acc = evaluate_predictions(validation_set, validation_labels, NBclassifier)

end = time.time()
print('Ran in {} seconds'.format(round(end - start, 3)))
