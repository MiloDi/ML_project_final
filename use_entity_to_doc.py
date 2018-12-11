import pandas as pd
df = pd.read_pickle('entity_to_doc.pkl')

# Specify the minimum entities to return
min_entities = 5
# Specify the document to evaluate
text_index = 9

# Specify size of corpus
corp_size = 20

# Gets the column of the doc we want
doc = df[text_index]

# Calcs the totals of each entity
total = df.sum(axis = 1)

# Used to store the frequencies of the entities in the given doc
ratios = {}

# Calculates what ratio of each entity appear in the given doc
for x in doc.iteritems():
	if total[x[0]] == 0:
		ratios[x[0]] = 0
	else:
		ratios[x[0]] = x[1]/total[x[0]]

# Sorts the entities in descending order of their frequencies
sorted_by_value = (sorted(ratios.items(), key=lambda kv: kv[1]))
sorted_by_value.reverse()

# returns the enitties
for counter, x in enumerate(sorted_by_value):
	if (counter <= min_entities-1 and x[1] > (1/corp_size)) or x[1] > (2/corp_size):
		print(x[0])
	


