import nltk
from functools import lru_cache

import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from bs4 import BeautifulSoup
import re


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stemmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=10000)(nltk.PorterStemmer().stem)
        self.tokenize = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text):

        # remove email addresses
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", ' ', str(text))
        # text = re.sub("[^a-z\s]+", " ", text)
        # text = re.sub("(\s+)", " ", text)
        # apply the Regexp tokenizer
        tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text)

        # Lower case normalisation
        tokens = [token.lower() for token in tokens]

        # Stop word removal
        tokens = [token for token in tokens if token not in self.stop_words]

        # Porter Stemming
        tokens = [self.stem(token) for token in tokens]

        # Pos tagged wordnet lemmatization
        # Note the tagged lemmatization takes 10minutes + to run
        tagged_tokens = pos_tag(tokens)
        tokens = []
        for token, tag in tagged_tokens:
            postag = tag[0].lower()
            if postag in ['r', 'a', 'v', 'n']:
                lemma = self.lemmatizer.lemmatize(token, postag)
            else:
                lemma = token
            tokens.append(lemma)

        return tokens
