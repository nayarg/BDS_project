import os
import pandas as pd
import numpy as np
import argparse
import ujson
import warnings
import matplotlib.pyplot as plt
import json

warnings.simplefilter("ignore")

""" CALCULATES RELVANCY FOR EACH SUBMISSION & PLOTS RELEVANCY HISTOGRAM """ 
""" ONLY OUTPUTS TOP 25% RELEVANT REVIEW_ID:RELEVANCY PAIRS """

def parse_arguments():

    parser = argparse.ArgumentParser(description="reads in .counts files")
    parser.add_argument('cluster', type=str, default='')
    parser.add_argument('counts', type=str, default='')
    parser.add_argument('title', type=str, default='')
    parser.add_argument('data_file', type=str, default='')
    parser.add_argument('master_data_file', type=str, default='')

    args = parser.parse_args()

    return args


def relevant_data(clusterFile, countFile, dataName, filePath):
    """
    :param clusterFile: cluster.csv
    :param countFile: clusters_freq_count.counts
    :return: list of relevant ids
    """
    data = []
    with open(filePath) as data_file:
        for line in data_file:
            data.append(json.loads(line))


    count = ujson.loads(open(countFile).read())
    cluster = pd.read_csv(clusterFile, header=None, delimiter=",", usecols=[0])
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

    #print(len(cluster_relevance_all))
    #print((cluster_relevance_all))
    # [0:(int)(len(cluster_relevance_all)*.25)]
    new_cluster_relevance_all=[(k,cluster_relevance_all[k]) for k in sorted(cluster_relevance_all, key=cluster_relevance_all.get,reverse=True)]
    idx=0
    new_cluster_relevance_all=new_cluster_relevance_all[0:(int)(len(cluster_relevance_all)*.25)]
    
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
##    print(len(out))


    ############################ KEY IS REVIEW ID #################################

    ujson.dump(top_reviewid_rel_pairs, open(os.path.join(os.path.dirname(countFile) + '/' + dataName + '_rel.json'), 'w'))
    ujson.dump(out, open(os.path.join(os.path.dirname(countFile) + '/' + dataName + '_final_output.json'), 'w'))
    return list_hist


def plot_histograms(data, title):

    val = np.array(data)
    plt.hist(val, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title(title)
    plt.xlabel('Percent Relevant')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":

    args = parse_arguments()
    cluster = args.cluster
    counts = args.counts
    title = args.title
    dataName = args.data_file
    inputData = args.master_data_file

    list_hist = relevant_data(cluster, counts, dataName, inputData)
    plot_histograms(list_hist, title)
