import torch.nn as nn

'''
This script defines the architecture of the LSTM model.
The class below intializes the embedding layer, LSTM layer, and fully connected layer.
The forward function defines the forward pass through the network and init_hidden intializes 
the hidden state of the LSTM.
'''

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        device = weight.device
        return (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device),
                weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device))
