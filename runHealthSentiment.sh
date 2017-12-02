#!/bin/bash

DataPath='dataset/sample.json'
BusinessDataPath='dataset/business.json'

OutputPath='test_output0'

NumClusters=100
Filter_Reviews_By_Relevency_Percentile=.35

python3 text_clean.py $DataPath $OutputPath $NumClusters --verbose
python3 filter_data.py $OutputPath

read -p "Manually label relevent clusters in first column of clusters.csv with 0 or 1, then Press [Enter] key to start parse_counts..."

python3 parse_counts.py $OutputPath $DataPath Filter_Reviews_By_Relevency_Percentile

python3 sentiment.py $OutputPath $BusinessDataPath