import os
from os.path import join
import numpy as np
import pickle
import argparse
import csv
import ujson
import time

"""
Reads in clusters.csv files
Reads in wordlists.list

Searches each cluster of words & determines cluster word count for each sentence

Outputs clusters_freq_count.counts


Run:
python3 filter_data.py "output"

"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Determines cluster word count for each sentence")
    parser.add_argument('input_file', type=str, default='')    
    args = parser.parse_args()

    return args


def load_cluster(clusterFile):
    clusters = []
    pathsList = []
    if os.path.isdir(clusterFile):
        for file in os.listdir(clusterFile):
            if os.path.splitext(file)[-1] == '.csv' or os.path.splitext(file)[-1] == '.tsv':
                pathsList.append(os.path.join(clusterFile, file))
    else:
        filePath = clusterFile.strip("[")
        filePath = filePath.strip("]")
        pathsList = (filePath).split(',')

    for path in pathsList:
        with open(path) as csvfile:
            clusterlist = csv.reader(csvfile)
            for row in clusterlist:
                clusters.append(row)
    return clusters


def main(input_file):
    clusters = load_cluster(join(input_file,'clusters.csv'))

    wordlist = pickle.load(open(join(input_file,'wordlists.list'), 'rb'))

    assert (isinstance(wordlist, dict))  # needs to be dict keyed with ids
    print("Loaded prior output files for Filter Data")
    num = len(wordlist)
    print("Data has {} sentences".format(num))

    print("Begin counting review words in cluster...")
    num_clusters = len(clusters)

    final_counts = {}
    progress = 0
    for post_id, sentences in wordlist.items():
        cluster_word_count = np.zeros((num_clusters,))
        for sentence in sentences:
            for word in sentence:
                for i in range(num_clusters):
                    if word in clusters[i]:
                        cluster_word_count[i] += 1

        sumcl = cluster_word_count.sum()
        progress += 1
        if sumcl != 0:
            cluster_word_count /= sumcl  # normalizes it
        if progress % 10000 == 0:
            print("completed counting: {:.2f}%".format((progress/num)*100))

        final_counts[post_id] = cluster_word_count

    print("Writing freq_counts output file ... ")
    ujson.dump(final_counts, open(join(input_file, 'clusters_freq_count.counts'), 'w'))



if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file

    start_time = time.time() #time in seconds
    
    main(input_file)

    total_time = (time.time() - start_time) 
    print('Filter Data Runtime (s): ' + str(total_time))
    print("Filter Data Complete \n")