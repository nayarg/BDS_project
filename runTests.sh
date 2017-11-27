#!/bin/bash

DataPath='../dataset/sample.json'
OutputPath='yelp_review'

#python3 text_clean.py $DataPath $OutputPath
python3 filter_data.py $OutputPath/TC_saveFolder/wordlists.list $OutputPath/clusters.csv
python3 parse_counts.py $OutputPath/clusters.csv $OutputPath/clusters_freq_count.counts 'Yelp Review Relevancy' $OutputPath $DataPath