import os
import pandas as pd
import argparse
import ujson
import json
from bs4 import BeautifulSoup
import re
import datetime
import warnings

warnings.simplefilter("ignore")


""" GETS INPUT READY FOR INDICIO """ 


def parse_arguments():

    parser = argparse.ArgumentParser(description="reads in original data files / directories")
    parser.add_argument('orgdata', type=str, default='')
    parser.add_argument('savedir', type=str, default='')
    parser.add_argument('reldata', type=str, default='')
    parser.add_argument('outputfile', type=str, default='')

    args = parser.parse_args()

    return args


def load_file(filePath, saveData, relData, outputFile):

    # file = pd.read_csv()
    fname, ext = os.path.splitext(filePath)


    dictionary = {}
    if ext == '.json':
        #data = ujson.loads(open(filePath).read())

        data = []
        with open(filePath) as data_file:
            for line in data_file:
                data.append(json.loads(line))
        #print (data)

        for d1 in data:
            bid = d1.get('business_id')
            review = d1.get('text')
            rid = d1.get('review_id')
            dict_temp = {bid : review}
            dictionary[rid] = dict_temp
    elif ext == '.csv' or ext == '.tsv':
        data = pd.read_csv(filePath, header=0, index_col=[], delimiter=",", quoting=1, encoding='latin1')
        for row in data.itertuples():
            if (not (pd.isnull(row.id) or pd.isnull(row.text))):
                dictionary[row.id] = row.text

    else:
        pathsList = []
        if os.path.isdir(filePath):
            for file in os.listdir(filePath):
                if os.path.splitext(file)[-1] == '.csv' or os.path.splitext(file)[-1] == '.tsv' or os.path.splitext(file)[-1] == '.txt':
                    pathsList.append(os.path.join(filePath, file))
        else:
            filePath = filePath.strip("[")
            filePath = filePath.strip("]")
            pathsList = (filePath).split(',')

        for path in pathsList:
            print("read: " + path)
            data = pd.read_table(path, header=0, delimiter="\t", encoding='latin1')
            rel = ujson.loads(open(relData).read())

            for row in data.itertuples():
                if (not (pd.isnull(row[1]) or pd.isnull(row[2]))):

                    cleantext = BeautifulSoup(row[2]).get_text()
                    cleantext = re.sub("'", "", cleantext)
                    cleantext = re.sub("[^a-zA-Z]", " ", cleantext)

                    try:
                        if rel.get(row[2]):
                            dictionary[row[2]] = [cleantext, str(shorten(row[1])), rel.get(row[2])]

                    except:
                        pass

    ujson.dump(dictionary, open(os.path.join(saveData + '/indi_input_' + outputFile + '.txt'), 'w'))
    return dictionary


def convertunixtodate(unix_time):

    dte = (datetime.datetime.fromtimestamp(int(unix_time))).strftime('%m-%d-%Y')
    return dte


def shorten(dtetime):

    dte = datetime.datetime.strptime(dtetime, '%m/%d/%y %H:%M').strftime('%m-%d-%Y')
    return dte


if __name__ == "__main__":

    args = parse_arguments()
    orgdata = args.orgdata
    savedir = args.savedir
    reldata = args.reldata
    outputfile = args.outputfile

    load_file(orgdata, savedir, reldata, outputfile)


