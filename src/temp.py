from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Sample text data
text = "I feel good but sometimes I feel not good."

# Tokenize the text
tokens = word_tokenize(text)

# Find bigrams
bigram_finder = BigramCollocationFinder.from_words(tokens)
bigrams = bigram_finder.nbest(BigramAssocMeasures().pmi, 10)
print(bigrams)
