import os
from os.path import join
import argparse
import pandas as pd
import csv
import json
import nltk
from nltk import tokenize
import numpy as np
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer



"""   
Reads in business.json from output folder
Reads in _final.json

Outputs 'business_scores_subset.csv' of Health Sentiment Rating per Restaurant

Run:
python3 sentiment.py "output" "dataset/business.json"

"""
def parse_arguments():

	parser = argparse.ArgumentParser(description="Takes input and data directory")
	parser.add_argument('input_file', type=str, default='')
	parser.add_argument('data_file', type=str, default='')
    args = parser.parse_args()

    return args


def sentiment(sentence):
	analyser = SentimentIntensityAnalyzer()
	ss = analyser.polarity_scores(sentence)
	#print ss.keys()
	return ss['compound']


def score_to_index(score):
	if (score < 0):
		oldmin = -1
		oldmax = -0.01
		newmin = 1
		newmax = 2
	elif (score >= 0 and score < 0.5):
		oldmin = 0
		oldmax = 0.49
		newmin = 2
		newmax = 3
	else:
		oldmin = 0.5
		oldmax = 1
		newmin = 3
		newmax = 5

	oldrange = (oldmax - oldmin)
	newrange = (newmax - newmin)


	index = (((score - oldmin) * newrange) / oldrange) + newmin
	dec = index % 1
	if (dec >= .25 and dec <=.75):
		round_index = np.floor(index) + .5
	else:
		round_index = int(round(index))
	return round_index

def sentiment_output(input_file, data_file):
	health_words = set()
	df_clusters_all = pd.read_csv(join(input_file,'clusters.csv'), header=None)
	#df_clusters_all = pd.read_csv('clusters.csv', header = 0)
	df_clusters_health = df_clusters_all.loc[df_clusters_all['label'] == 1]
	df_clusters_health_nolabel = df_clusters_health.drop('label', axis=1)

	for row in df_clusters_health_nolabel.iterrows():
		temp = []
		index, data = row

		for word in data:
			if (type(word) != float):
				#print word
				health_words.add(word)

	if ('nan' in health_words):
		health_words.remove('nan')

	health_words_string = ' '.join(health_words)
	#print health_words_string

	bus_score_dict = {}
	bus_score_count = {}
	rev_score_dict = {}
	rev_score_count = {}

	with open(join(input_file,'final_output.json')) as rev_data:
		data = json.load(rev_data)
		for rev_id in data:
			dict_b_r = data[rev_id]
			#print dict_b_r.keys()
			for bus_id in dict_b_r:
				review = dict_b_r[bus_id]
				rev_sentences = tokenize.sent_tokenize(review)
				for sentence in rev_sentences:
					count = 0
					sent_list = sentence.split()
					for word in sent_list:
						if (word in health_words):
							count = count + 1
					if (count > 0):
						score = sentiment(sentence)

						if (bus_id in bus_score_dict):
							old_score = bus_score_dict[bus_id]
							bus_score_dict[bus_id] = old_score + score

							bus_score_count[bus_id] = bus_score_count[bus_id] + 1
						else:
							bus_score_dict[bus_id] = score

							bus_score_count[bus_id] = 1

						if (rev_id in rev_score_dict):
							old_score = rev_score_dict[rev_id]
							rev_score_dict[rev_id] = old_score + score

							rev_score_count[rev_id] = rev_score_count[rev_id] + 1
						else:
							rev_score_dict[rev_id] = score

							rev_score_count[rev_id] = 1

	for bus_id in bus_score_dict:
		bus_score_dict[bus_id] = bus_score_dict[bus_id] / bus_score_count[bus_id]

	for rev_id in rev_score_dict:
		rev_score_dict[rev_id] = rev_score_dict[rev_id] / rev_score_count[rev_id]

	#print ('business scores')
	#print bus_score_dict

	#print ('review scores')
	#print rev_score_dict

	bus_dict = {}
	name_score_dict = {}
	name_index_dict = {}

	with open(data_file, 'r') as bus_names:
		bus_data = [json.loads(line) for line in bus_names]

		for bus in bus_data:
			bus_dict[bus["business_id"]] = bus["name"]

		for bus_id in bus_score_dict:
			name = bus_dict[bus_id]
			score = bus_score_dict[bus_id]
			name_score_dict[name] = score

			index = score_to_index(score)
			name_index_dict[name] = index

	#print name_score_dict
	#print name_index_dict

	with open(join(input_file,'business_scores_subset.csv'), 'wb') as f:
		w = csv.writer(f)
		w.writerow(['Business Name', 'Health Score', 'Health Index'])
		for key, value in name_score_dict.items():
			try:
				w.writerow([key, value, name_index_dict[key]])
			except UnicodeEncodeError:
				print ('non ascii')
				pass


if __name__ == "__main__":

    args = parse_arguments()
    
    start_time = time.time() #time in seconds

    input_file = args.input_file
    data_file = args.data_file

    sentiment_output(input_file, data_file)

    total_time = (time.time() - start_time) 
    print('Sentiment Runtime (s): ' + str(total_time))    



