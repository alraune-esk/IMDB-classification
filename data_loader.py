"""This file contains all of the code for
reading data and converting it to the appropriate form for models to use."""
from collections import Counter, defaultdict
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import __settings__
from preprocessor import Preprocessor


class TextDS:
    def __init__(self, path, preprocessor=Preprocessor(), num_terms=10000, batch_size=__settings__.batch_size, split=__settings__.split, window_size=__settings__.window_size, dev='cpu'):
        """
        :param path: Path to dataset .csv file.
        :param preprocessor: Specifies how to split strings into lists of tokens.
        :param num_terms: The dataset only gives distinct IDs to the num_terms most frequent terms, all other terms are given a special ID for "other".
        :param batch_size: The number of documents that will be provided in each batch.
        :param split: A tuple that specifies how to split data into train, validation and test sets. First number
        is proportion of data in training set, second number is proportion of data in validation set, all reamining
        data is in test set.
        :param window_size: For word2vec mode, specifies the (half) context window size, so the window is given from
        i-window_size up to i+window_size.
        """
        self.df = pd.read_csv(path).sample(frac=1.0, random_state=6490)
        self.batch_size = batch_size
        self.dev = dev
        self.window_size = window_size
        self.num_terms = num_terms

        self.apply_preprocessor(preprocessor)
        self.split(*split)

    def apply_preprocessor(self, preprocessor):
        self.df['tokens'] = [preprocessor(s) for s in self.df['text']]
        self.token_to_count = Counter([x for l in self.df['tokens'] for x in l])
        self.vocab = list([term for term, count in self.token_to_count.most_common(self.num_terms)])
        self.token_to_id = {self.vocab[i]: i+2 for i in range(len(self.vocab))}
        self.df['ids'] = [[self.token_to_id.get(t, 1) for t in sentence] for sentence in self.df['tokens']]

    def split(self, train, valid):
        self.train_index = int(train * len(self.df))
        self.valid_index = int(valid * len(self.df)) + self.train_index
        self.train = slice(0, self.train_index)
        self.valid = slice(self.train_index, self.valid_index)
        self.test = slice(self.valid_index, len(self.df))

    def set_partition(self, data_split):
        self.partition = data_split

    def shuffle(self):
        self.df.iloc[self.train] = self.df.iloc[self.train].sample(frac=1.0)

    def get_batch(self, index):
        batch = self.df.iloc[self.partition].iloc[index:index+self.batch_size]
        x, y = self.batch_method(batch)
        return torch.LongTensor(x).to(self.dev), torch.LongTensor(y).to(self.dev)

    def batch_method(self, batch):
        """
        Gets one W2V batch.
        Converts all of the sentences in batch into one batch of context windows,
        with the middle word of each window as target.
        """
        w = self.window_size  # The number of words on either side of the target (centre) word in the context window.
        sentences = list(batch['ids'].values)  # sentences is a list of all of the sentences in the batch (in word ID form).

        x = []
        y = []
        # iterate through each sentence and return x containing a list of word ids from the i'th context window
        # and y a list of centre word ids.
        for sentence in sentences:
            if len(sentence) > 2 * w:
                counter = w
                # get the centre word and the words surrounding it according to the window size.
                while counter + w < len(sentence):
                    context_window = sentence[counter - w: counter]
                    context_window.extend(sentence[counter + 1: counter + w + 1])
                    centre_word = sentence[counter]
                    counter += 1
                    y.append(centre_word)
                    x.append(context_window)
        return x, y

    def get_batches(self):
        return (self.get_batch(i) for i in range(0, self.partition.stop - self.partition.start, self.batch_size))


class LabelledTextDS(TextDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        classes = self.df['label'].unique()
        self.class_to_id = {classes[i]: i for i in range(len(classes))}
        self.df['label'] = [self.class_to_id[c] for c in self.df['label']]

    def pad_batch(self, ids):
        """
        Pads each sentence in ids with 0 tokens so that they all have the same length (length of the longest sentence).
        """
        max_len = max([len(x) for x in ids])
        return [x + [0] * (max_len - len(x)) for x in ids]

    def batch_method(self, batch):
        x = self.pad_batch(batch['ids'])
        y = batch['label'].values
        return x, y

    def get_vector_representation(self):
        """
        This function converts the documents to tf-idf vectors and returns a sparse matrix representation of the data.
        You can change any of the settings of CountVectorizer.
        """
        vectorizer = CountVectorizer(lowercase=False,
                                     tokenizer=lambda x: x,  # Tokenization should already be done by preprocessor
                                     stop_words=None,
                                     min_df=5,
                                     max_features=None,  ## use all features
                                     ngram_range=(1, 1),  ## This uses only unigram counts
                                     binary=False)  ## This sets the beatures to be frequency counts
        pipeline = Pipeline([('vec', vectorizer), ('tfidf', TfidfTransformer())])

        X = pipeline.fit_transform(self.df['tokens'])
        Y = self.df['label'].values
        return ((X[:self.train_index], Y[:self.train_index]),
                (X[self.train_index:self.valid_index], Y[self.train_index:self.valid_index]),
                (X[self.valid_index:], Y[self.valid_index:]))