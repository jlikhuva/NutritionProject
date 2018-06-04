import torch
import random
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
        dropout_keep_prob=1, max_length=110, num_layers=4
    ):
        super(DecoderNet, self).__init__()
        self.embed = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(
            2*embed_size, hidden_size, num_layers, batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.max_length = max_length
        self.word_vectors = word_vectors
        self.output_size = output_size
        self._init()

    def _init(self):
        self.embed.weight.data = self.word_vectors

    # Foward and sample stolen from https://github.com/yunjey/pytorch-tutorial
    def forward(self, features, captions, lengths, teacher_forcing_ratio=0.5):
        """Decode image feature vectors and generate captions."""
        # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        loss = 0.0;
        embeddings = self.embed(captions)
        features = features.unsqueeze(1).expand(embeddings.shape)
        embeddings = torch.cat((features, embeddings), -1)

        states = None; weights = torch.ones(self.output_size); weights[622] = 0
        for t in range(embeddings.shape[1]):
            use_teacher_forcing = True
            if t == 0: inputs = embeddings[:, t, :].unsqueeze(1)
            hiddens, states = self.gru(inputs, states)
            cur_output = self.linear(hiddens.squeeze(1))
            _, predicted = cur_output.max(1)

            loss_function = nn.CrossEntropyLoss(weight=weights, reduce=False)
            if not use_teacher_forcing:
                predicted = predicted.detach()
                inputs = self.embed(predicted).unsqueeze(1)
                inputs = torch.cat((features[:, t, :].unsqueeze(1), inputs), -1)
                loss += loss_function(cur_output, captions[:, t])
            else:
                inputs = embeddings[:, t, :].unsqueeze(1)
                loss += loss_function(cur_output, captions[:, t])
        return loss


    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        START = self.word_vectors[621].expand((features.shape[0], 1, 100))
        inputs = START.to(device)
        features = features.unsqueeze(1).expand(inputs.shape)
        embeddings = torch.cat((features, inputs), -1)

        for i in range(self.max_length):
            hiddens, states = self.gru(embeddings, states)       # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            embeddings = self.embed(predicted).unsqueeze(1)      # inputs: (batch_size, embed_size)
            embeddings = torch.cat((features, embeddings), -1)   # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_length)
        return sampled_ids
