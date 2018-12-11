import glob
import spacy
import errno
import numpy
import pandas
import nltk
import random
import collections
import scipy
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from pathlib import Path
import sklearn
#nlp= spacy.load('en_core_web_sm')
#python -m spacy download en_core_web_sm #IF ERROR

print(" ")
print(" ")
print("BEGIN")

#Code below found at: https://stackoverflow.com/questions/41002041/read-multiple-txt-files-in-a-single-folder
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

#stop words condition
stop_words = stopwords.words('english')
stop_words.extend(['the','of', 'as', 'is'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

word_list = list(sent_to_words(text_file_list))

#remove stop words
for text_file in word_list:
    for word in text_file:
        if word in stop_words:
            text_file.remove(word)

#run train test split on text file list #THIS ISN"T GOING TO WORK
file_train, file_test  = train_test_split(text_file_list)

#create the vectorized bag of words version of the corpus
counter = CountVectorizer(stop_words = "english", ngram_range=(1,3))
X_train_bow = counter.fit_transform(file_train)
X_test_bow = counter.transform(file_test)
#TFIDF vectorizer
tfidfer = TfidfTransformer()
X_train_tfidf = tfidfer.fit_transform(X_train_bow)
X_test_tfidf = tfidfer.transform(X_test_bow)

##INPUT IN CORPUS
#Query string, this will likely only work if the words are in the training corpus
input_string = "virus poses as christmas"
print("Input:", input_string, "\n")
input_bow = counter.transform([input_string])
input_tfidf = tfidfer.transform(input_bow)

sims = numpy.array([sklearn.metrics.pairwise.cosine_similarity(X_train_tfidf[i], input_tfidf) for i in range(X_train_tfidf.shape[0])]).flatten()
k = 3 # Select k most similar docs
bestK = sims.argsort()[-k:][::-1]
print("Best document from TFIDF for the query that is in the corpus: \n")
print(file_train[bestK[0]])
print(f"Similarity score: {sims[bestK[0]]} \n")

##INPUT NOT IN CORPUS
#Query string, this will likely only work if the words are in the training corpus
input_string = "vicarious"
print("Input:", input_string, "\n")
input_bow = counter.transform([input_string])
input_tfidf = tfidfer.transform(input_bow)

sims = numpy.array([sklearn.metrics.pairwise.cosine_similarity(X_train_tfidf[i], input_tfidf) for i in range(X_train_tfidf.shape[0])]).flatten()
k = 3 # Select k most similar docs
bestK = sims.argsort()[-k:][::-1]
print("Best document from TFIDF for the query that is NOT in the corpus: \n")
print(file_train[bestK[0]])
print(f"Similarity score: {sims[bestK[0]]} \n")


#DOC_2_VEC
#https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
print(" ")
print("Now using the Doc2Vec model")
print(" ")
def train_test_split(sentences,  tokens_only=False):
    i=0
    for sentence in sentences:       
        if tokens_only:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(str(sentence), deacc=True), [i])
            i+=1
#Create the training and testing data, the test documents need to be tagged (see link above)
train_corpus = list(train_test_split(text_file_list))
test_corpus = list(train_test_split(text_file_list, tokens_only=True))

#build and train the doc2vec model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#FOR SIMILARITY BETWEEN STRING(IN CORPUS) AND DOCUMENTS
#find a query search, https://stackoverflow.com/questions/42781292/doc2vec-get-most-similar-documents
input_string = "virus poses as christmas" #query string, this will likely only work if the words are in the training corpus
print("Input that is in the corpus:", input_string)
print("")
query_vector = model.infer_vector(input_string) 
sims = model.docvecs.most_similar([query_vector])

#print out the most similar and its score
for label, index in [('Best document from DOC2VEC for the query that is in the corpus: \n', 0), ('Second best', 1)]:
    print("")
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


#FOR SIMILARITY BETWEEN STRING(NOT IN CORPUS) AND DOCUMENTS
input_string = "vicarious".split() #query string, this will likely only work if the words are in the training corpus
print("Input that is not in the corpus:", input_string)
print("")
query_vector = model.infer_vector(input_string) 
sims = model.docvecs.most_similar([query_vector])

#print out the most similar and its score
for label, index in [('Best document from DOC2VEC for the query that is NOT in the corpus:\n', 0), ('Second best', 1)]:
    print("")
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))