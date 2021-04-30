import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        """
        :param vocab_size: the number of different embeddings to make (need one embedding for every unique word).
        :param embedding_dim: the dimension of each embedding vector.
        :param num_classes: the number of target classes.
        :param word_embeddings: optional pre-trained word embeddings. If not given word embeddings are trained from
        random initialization. If given then provided word_embeddings are used and the embeddings are not trained.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.W = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """
        # Question 2 embedding based classifier
        # embed input ids to create word representations
        embedded_vectors = self.embeddings(x)
        # sum aggregation
        add_embeds = torch.sum(embedded_vectors, dim=1)
        # run it through the linear layer
        out = self.W(add_embeds)
        return out


class LSTMFastText_PosEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=40,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.pos_encoder = PosEncoding(embedding_dim)
        self.linear = nn.Linear(40, 40)
        self.linear2 = nn.Linear(40, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """
        # embed the word ids to get word representations
        embedded_vectors = self.embeddings(x)
        # add pos encoding to each vector
        embedded_vectors = self.pos_encoder(embedded_vectors)
        # run the vectors through a LSTM
        lstm_out, (ht, ct) = self.lstm(embedded_vectors)
        # run the last hidden state through linear layer
        output = F.relu(self.linear(ht[-1]))
        output = self.linear2(output)

        return output


class MultiLayerFastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        """
        :param vocab_size: the number of different embeddings to make (need one embedding for every unique word).
        :param embedding_dim: the dimension of each embedding vector.
        :param num_classes: the number of target classes.
        :param word_embeddings: optional pre-trained word embeddings. If not given word embeddings are trained from
        random initialization. If given then provided word_embeddings are used and the embeddings are not trained.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        # 2 layer neural net best performing settings: 40 neurons 2 layers 40 -> 40 -> 2.
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, num_classes)
        # self.fc2 = nn.Linear(embedding_dim//2, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """
        # embed input ids to get vector representation
        embedded_vectors = self.embeddings(x)
        # sum aggregation and relu activation.
        add_embeds = torch.sum(embedded_vectors, dim=1)
        out = F.relu(self.fc(add_embeds))
        out = self.fc1(out)
        # out = self.fc2(out)

        return out


class LSTMFastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=40,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(40, 40)
        self.linear2 = nn.Linear(40, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """
        # embed word ids to get vector representations
        embedded_vectors = self.embeddings(x)
        # run the vectors through a lstm
        lstm_out, (ht, ct) = self.lstm(embedded_vectors)
        # run the last hidden state through 2 linear layers
        output = F.relu(self.linear(ht[-1]))
        output = self.linear2(output)

        return output


class PosEncoding(nn.Module):

    def __init__(self, dim):
        super(PosEncoding, self).__init__()
        # p = 0.1
        self.dropout = nn.Dropout(0.1)

        # set max to be 1000
        pos_encoding = torch.zeros(1000, dim)

        # as per the formula for positional encoding:
        # pe(pos, 2i)  = sin(pos / denom), pe(pos, 2i+1) = cos(pos / denom).
        denom = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        pos_encoding[:, 0::2] = torch.sin(position * denom)
        pos_encoding[:, 1::2] = torch.cos(position * denom)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        # prevent SGD from updating the pos encodings.
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # add the pos encoding
        x = x + self.pos_encoding[:x.size(0), :]
        return self.dropout(x)
