import os
from os.path import join
import pandas as pd
import numpy as np
import argparse
import ujson
import warnings
import matplotlib.pyplot as plt
import json
import time

warnings.simplefilter("ignore")

plot = False

"""   
Reads in data.json
Reads in clusters.csv
Reads in clusters_freq_count.counts

Outputs _final.json for sentiment of top X*100% reviews

Run:
python3 parse_counts.py "output" "dataset/sample.json" .35

If plot wanted:
python3 parse_counts.py "output" "dataset/sample.json" .35 --plot
"""

def parse_arguments():

    parser = argparse.ArgumentParser(description="Input, data directories and top review percentage (.xx)")
    parser.add_argument('input_file', type=str, default='')
    parser.add_argument('data_file', type=str, default='')
    parser.add_argument('percent', type=str, default='')
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    return args


def relevant_data(input_file, data_file, percentage):
    data = []
    with open(data_file) as data_file:
        for line in data_file:
            data.append(json.loads(line))

    percent = float(percentage)

    count = ujson.loads(open(join(input_file,'clusters_freq_count.counts')).read())
    cluster = pd.read_csv(join(input_file,'clusters.csv'), header=None, delimiter=",", usecols=[0])
    clusterlist = cluster[0].tolist()
    clusters_score = np.array(clusterlist)

    # dictionary
    cluster_relevance_all = {}
    list_hist = []
    for id, cluster_list in count.items():

        relevancy = np.array(cluster_list)
        count_relevancy = clusters_score*relevancy
        sum_rel = sum(count_relevancy)
        cluster_relevance_all[id] = sum_rel
        list_hist.append(sum_rel)

    new_cluster_relevance_all=[(k,cluster_relevance_all[k]) for k in sorted(cluster_relevance_all, key=cluster_relevance_all.get,reverse=True)]
    idx=0
    new_cluster_relevance_all=new_cluster_relevance_all[0:(int)(len(cluster_relevance_all)*percent)]
    
    top_reviewid_rel_pairs={}
    out={}

    for k,v in new_cluster_relevance_all:
        top_reviewid_rel_pairs[k]=v
        for dictx in data:
            if dictx['review_id'] == k:
                bus = dictx['business_id']
                text = dictx['text']
                out[k] = {}
                out[k][bus] = text
                break

    # KEY IS REVIEW ID, SUBKEY IS BUSINESS ID
    ujson.dump(out, open(join(input_file,'final_output.json'), 'w'))

    return list_hist


def plot_histograms(data, percentage):
    val = np.array(data)
    plt.hist(val, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Top ' + str(percentage)[1:] + '% Reviews')
    plt.xlabel('Percent Relevant')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":

    args = parse_arguments()
    
    start_time = time.time() #time in seconds

    input_file = args.input_file
    data_file = args.data_file
    percentage = args.percent
    plot =args.plot


    list_hist = relevant_data(input_file, data_file, percentage)

    total_time = (time.time() - start_time) 
    print('Parse Counts Runtime (s): ' + str(total_time))   

    if plot:
        plot_histograms(list_hist, percentage)
