import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class DecoderNet(nn.Module):
    def __init__(
        self, word_vectors, output_size, embed_size, hidden_size=1024,
        dropout_keep_prob=1, max_length=110, num_layers=3
    ):
        super(DecoderNet, self).__init__()
        self.embed = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(
            2*embed_size, hidden_size, num_layers, batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.max_length = max_length
        self.word_vectors = word_vectors
        self._init()

    def _init(self):
        self.embed.weight.data = self.word_vectors

    # Foward and sample stolen from https://github.com/yunjey/pytorch-tutorial
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generate captions."""
        embeddings = self.embed(captions)
        # print('Embeddings before = ', features.shape, embeddings.shape)
        features = features.unsqueeze(1).expand(embeddings.shape)
        embeddings = torch.cat((features, embeddings), -1)
        # print('Embeddings foward = ', embeddings.shape)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.gru(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        START = self.word_vectors[621].expand((features.shape[0], 1, 100))
        inputs = START.to(device)
        features = features.unsqueeze(1).expand(inputs.shape)
        embeddings = torch.cat((features, inputs), -1)

        for i in range(self.max_length):
            hiddens, states = self.gru(embeddings, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            embeddings = self.embed(predicted).unsqueeze(1)                       # inputs: (batch_size, embed_size)
            embeddings = torch.cat((features, embeddings), -1)      # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_length)
        return sampled_ids
