
import os
from os.path import join
import time
import pandas as pd
import numpy as np
import pickle
import warnings
import argparse
import logging
import csv
import ujson
import sys
import json

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk.data

from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
import scipy.spatial.distance as dist

warnings.simplefilter("ignore")
output_dir = 'output'
input_file = 'dataset/sample.json'
num_clusters = 100

verbose = False
np.random.seed(0)


""" 
Reads in JSON data file 

Parses text and clusters using Word2Vec

Outputs clusters.csv files
Outputs wordlists.list

Run:
python3 text_clean.py "dataset/sample.json" "sample_output" 10 --verbose

Future Experiments:
Can change word2vec vector dimensionality

"""



def parse_arguments():
    parser = argparse.ArgumentParser(description="Input, output directories and number of clusters for experiments")
    parser.add_argument('input_file', type=str, default=input_file)
    parser.add_argument('output_dir', type=str, default=output_dir)
    parser.add_argument('num_clusters', type=int, default=num_clusters)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    args = parser.parse_args()

    return args



def load_file(filePath):
    fname, ext = os.path.splitext(filePath)

    dictionary = {}
    data = []
    with open(filePath) as data_file:
        for line in data_file:
            data.append(json.loads(line))

    for d1 in data:
        bid = d1.get('business_id')
        review = d1.get('text')
        rid = d1.get('review_id')
        dict_temp = {bid : review}
        dictionary[rid] = dict_temp
        #print (dictionary)

    return dictionary

def get_CleanText(data, saveAs='', remove_stopwords=False):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    num_reviews = len(data)
    
    print("Creating wordlists and sentences")
    sentences = {}  # Initialize an empty dict of sentences which is keyed with the id
    wordlists = {}
    for id, bid_review in data.items():
        for bid, review  in bid_review.items():
            raw_sentences = tokenizer.tokenize(str(review).strip())

            # Loop over each sentence in review
            temp_sentences = []
            for raw_sentence in raw_sentences:
                # If a sentence is empty, skip it
                if len(raw_sentence) > 0:
                    # Get a list of words        
                    review_text = BeautifulSoup(raw_sentence).get_text()
                    # Remove non-letters
                    review_text = re.sub("'", "", review_text)
                    review_text = re.sub("[^a-zA-Z]", " ", review_text)
                    # Convert words to lower case and split them
                    temp_sentences += [[review_text.lower().split()]]

            sentences[id] = temp_sentences

            # Same parsing but for overall review to output wordlists
            review_text = BeautifulSoup(review).get_text()
            review_text = re.sub("'", "", review_text)
            review_text = re.sub("[^a-zA-Z]", " ", review_text)
            wordlists[id] = [review_text.lower().split()]

    pickle.dump(wordlists, open(saveAs, 'wb'))

    return sentences


def get_WordVecs(sentences, saveAs='', pretrained=False):

    if verbose: print("Calculating word vectors...")

    if isinstance(sentences, dict):
        sentence_list = []
        for sentence in sentences.values():
            for word in sentence:
                sentence_list += word
        sentences = sentence_list

    num_features = 300  # Word vector dimensionality
    min_word_count = max(20.0, len(sentences) * 0.00005)  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-4  # Downsample setting for frequent words

    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1, )

    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # save the model 
    words = set(model.wv.index2word)
    dictionary = {}
    for word in words:
        dictionary[word] = model.wv[word]

    return dictionary



def get_Clusters(word_vectors, n_clusters, saveAs=''):

    if verbose: print("Calculating word clusters...")

    keys = sorted(word_vectors.keys())

    items = sorted(word_vectors.items())
    N = len(items)
    M = len(items[0][1])
    matrix = np.zeros((N, M))

    for i in range(N):
        matrix[i, :] = items[i][1]

    # Initalize a k-means object and use it to extract centroids
    if verbose: print("Running K means...")
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, verbose=0)
    kmeans.fit(matrix)
    if verbose: print("Finished K means in " + str((time.time() - start)) + " seconds.")

    idx = kmeans.predict(matrix)
    word_centroid_map = dict(zip(keys, idx))

    # Print the first ten clusters
    all_words = []
    for cluster in range(n_clusters):
        # Find all of the words for that cluster number, and print them out
        words = []
        items = list(word_centroid_map.items())
        for i in range(len(items)):
            if (items[i][1] == cluster):
                words.append(items[i][0])

        all_words.append(words)

    if verbose: print("Sorting the clusters")

    sorted_clusters = []
    for i in range(len(all_words)):
        distances = []
        for word in all_words[i]:
            distances += [dist.euclidean(word_vectors[word], kmeans.cluster_centers_[i])]
        sorted_clusters.append([words for (dists, words) in sorted(zip(distances, all_words[i]))])

    with open(join(output_dir, 'clusters.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(sorted_clusters)

    pass



if __name__ == "__main__":
    args = parse_arguments()
    output_dir = args.output_dir
    num_clusters = args.num_clusters
    verbose = args.verbose
    input_file = args.input_file

    # For experimental runs
    start_time = time.time() 
    
    # Output Directory, make sure it doesn't already exist!
    os.makedirs(output_dir)

    # Reads in file as dict
    if verbose:
        print("Loading Dataset for Text Clean...")
    filtered_text = load_file(input_file)

    # Converts text to sentences dict for word2vec and writes wordlists.lists for filter_data.py  
    sentences = get_CleanText(filtered_text, saveAs=join(output_dir, 'wordlists.list'))
    if verbose:
        print("Length of sentences: " + str((len(sentences))))        


    # Calculates word_vectors using word2vec
    word_vectors = get_WordVecs(sentences, pretrained=False)
    if verbose:
        print("Length of word_vectors: " + str((len(word_vectors))))

    # Calculates word clusters using KMeans, main
    get_Clusters(word_vectors, num_clusters)

    total_time = (time.time() - start_time) 
    print('Text Clean Runtime (s): ' + str(total_time))
    print("Text Clean Complete \n")