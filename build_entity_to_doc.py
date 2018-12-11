import spacy
import glob
import string
from pathlib import Path
import pandas as pd

nlp = spacy.load('en_core_web_sm')

files = sorted(glob.glob(str( Path.cwd() / 'TEST' / '*.txt'))) #Filepath to current working directory /TESTS/*.txt
text_file_list = [] #this is the list that will hold all of the files as srings
text_names = [] #this conatins a list of the file paths to each text file
for name in files:
    text_names.append(name)
    try:
        with open(name) as f:
            temp_string = ""
            lines = f.read().splitlines()
            for line in lines:
                temp_string += line
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
    text_file_list.append(temp_string)

# Fills the corpus
corpus = ""
for x in text_file_list:
	corpus += x

table = str.maketrans({key: None for key in string.punctuation})
corpus = corpus.translate(table)
corp_doc = nlp(corpus)

corp_entities = {}
for entity in corp_doc.ents:
	if entity.label_ == "DATE" or entity.label_ == "TIME" or entity.label_ == "PERCENT" or entity.label_ == "MONEY" or entity.label_ == "QUANTITY" or entity.label_ == "ORDINAL" or entity.label_ == "CARDINAL":
		continue
	corp_entities[entity.text] = 0

corpus_list = corpus.split()
for x in corpus_list:
  corp_entities[x] = corpus_list.count(x)


# Builds and stores a dataframe
df = pd.DataFrame()

# Makes a list of all entities to be used as row indecis
all_ents = []
for x in corp_entities:
	all_ents.append(x)

# Counts the occurence of each entity in each doc
# and stores the results in a dataframe
for count, x in enumerate(text_file_list):
	doc_nums = []
	x = x.translate(table)
	x_doc = nlp(x)
	x_entities = {}
	for entity in x_doc.ents:
		if entity.label_ == "DATE" or entity.label_ == "TIME" or entity.label_ == "PERCENT" or entity.label_ == "MONEY" or entity.label_ == "QUANTITY" or entity.label_ == "ORDINAL" or entity.label_ == "CARDINAL":
			continue
		x_entities[entity.text] = 0

	x_list = x.split()

	for a in x_entities:
		x_entities[a] = x_list.count(a)
	for i in all_ents:
		if i in x_entities:
			doc_nums.append(x_entities[i])
		else:
			doc_nums.append(0)
	df[count] = pd.Series(doc_nums, index = all_ents)
	

# stores the dataframe in a file
df.to_pickle('entity_to_doc.pkl')

