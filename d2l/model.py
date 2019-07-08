"""The model module contains neural network building blocks"""
# import d2l
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['corr2d', 'linreg', 'RNNModel', 'MLPAttention', 'Seq2SeqEncoder', 'Encoder', 'Decoder', 'EncoderDecoder', 'masked_softmax']

def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def linreg(X, w, b):
	"""Linear regression."""
	return torch.mm(X,w) + b

class RNNModel(nn.Module):
    """RNN model."""

    def __init__(self, rnn_layer, num_inputs, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.Linear = nn.Linear(num_inputs, vocab_size)

    def forward(self, inputs, state):
        """Forward function"""
        X = F.one_hot(inputs.long().transpose(0,-1), self.vocab_size)
        X = X.to(torch.float32)
        state = state.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.Linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, num_hiddens, device, batch_size=1):
        """Return the begin state"""
        return torch.zeros(size=(1, batch_size, num_hiddens), dtype=torch.int64, device=device)
        
class Residual(nn.Module):
  
  def __init__(self,input_channels, num_channels, use_1x1conv=False, strides=1, **kwargs):
    super(Residual, self).__init__(**kwargs)
    self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
    self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    if use_1x1conv:
      self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)
    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, X):
    
    Y = self.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3:
      X = self.conv3(X)
    Y += X
    Y =self.relu(Y)
    return Y

class MLPAttention(nn.Module):  
    def __init__(self, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # Use flatten=True to keep query's and key's 3-D shapes.
        self.W_k = nn.Linear(units, units, bias=False)
        self.W_q = nn.Linear(units, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length):
        print(query.shape,'asdfghj')
        query, key = torch.tanh(self.W_k(query)), torch.tanh(self.W_q(key))
        # expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast.
        features = query.unsqueeze(2) + key.unsqueeze(1)
        scores = self.v(features).squeeze(-1)
        # print(scores, valid_length)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        # print(attention_weights)
        return torch.bmm(attention_weights, value)

# class Seq2SeqEncoder(nn.Module):
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout=0, **kwargs):
#         super(Seq2SeqEncoder, self).__init__(**kwargs)
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = nn.LSTM(num_hiddens, num_layers, dropout=dropout)

#     def forward(self, X, *args):
#         # print(X)
#         X = self.embedding(X) # X shape: (batch_size, seq_len, embed_size)
#         # print(X.shape)
#         X = X.permute(1, 0, 2)  # RNN needs first axes to be time
#         state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.context)
#         out, state = self.rnn(X, state)
#         # The shape of out is (seq_len, batch_size, num_hiddens).
#         # state contains the hidden state and the memory cell
#         # of the last time step, the shape is (num_layers, batch_size, num_hiddens)
#         return out, state

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size,num_hiddens, num_layers, dropout=dropout)
   
    def begin_state(self, batch_size, device):
        return [torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device),
                torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device)]
    def forward(self, X, *args):
        X = self.embedding(X) # X shape: (batch_size, seq_len, embed_size)
        X = X.transpose(0, 1)  # RNN needs first axes to be time
      #  state = self.begin_state(X.shape[1], device=X.device)
        out, state = self.rnn(X)
        # The shape of out is (seq_len, batch_size, num_hiddens).
        # state contains the hidden state and the memory cell
        # of the last time step, the shape is (num_layers, batch_size, num_hiddens)
        return out, state

class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        """Forward function"""
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder archtecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        """Return the begin state"""
        raise NotImplementedError

    def forward(self, X, state):
        """Forward function"""
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """Forward function"""
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    # print(X.shape)
    if valid_length is None:
        return F.softmax(X, dim=0)
    else:
        shape = X.shape
        if valid_length.ndim == 1:
            valid_length = valid_length.repeat(shape[1], axis=0)
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = nd.SequenceMask(X.reshape((-1, shape[-1])), valid_length, True,
                            axis=1, value=-1e6)
        return X.softmax().reshape(shape)