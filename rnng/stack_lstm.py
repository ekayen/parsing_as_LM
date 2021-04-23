""" 
Original code by Huimeng Zhang:
https://github.com/tianlinyang/stack-lstm-ner/blob/master/model/stack_lstm.py

Adapted by Elizabeth Nielsen
"""

class StackRNN(object):
    def __init__(self, cell, dropout, get_output):
        self.cell = cell
        self.dropout = dropout
        self.get_output = get_output

    def initialize_tensors(self,initial_state,p_empty_embedding=None):
        self.s = [(initial_state, None)]
        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding
    
    def push(self, expr, extra=None):
        self.dropout(self.s[-1][0][0])
        self.s.append((self.cell(expr, self.s[-1][0]), extra))

    def pop(self):
        return self.s.pop()[1]

    def embedding(self):
        return self.get_output(self.s[-1][0]) if len(self.s) > 1 else self.empty

    def back_to_init(self):
        while self.__len__() > 0:
            self.pop()

    def clear(self):
        self.s.reverse()
        self.back_to_init()

    def __len__(self):
        return len(self.s) - 1


from torch import nn
import torch

class RNNG(nn.Module):
    def __init__(self,dropout_ratio=0.2,tok_emb_dim=300,lstm_hid_dim=256,batch_size=1,output_dim=2,vocab_size=10):
        super().__init__()
        self.tok_emb_dim = tok_emb_dim
        self.dropout_ratio = dropout_ratio
        self.lstm_hid_dim = lstm_hid_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        
        self.use_gpu = False#torch.cuda.is_available()
        
        self.dropout = nn.Dropout(p=self.dropout_ratio)

        self.tok_emb = nn.Embedding(self.vocab_size,self.tok_emb_dim)
        self.lstm_cell = nn.LSTMCell(input_size=self.tok_emb_dim, hidden_size=self.lstm_hid_dim)
        self.lstm = nn.LSTM(input_size=self.tok_emb_dim, hidden_size=self.lstm_hid_dim)

        self.linear = nn.Linear(self.lstm_hid_dim,self.output_dim)

        self.stack = StackRNN(self.lstm_cell,
                              self.dropout,
                              self._rnn_get_output)
        
    def _rnn_get_output(self,state):
        return state[0]

    @staticmethod
    def xavier_init(gpu, *size):
        return nn.init.xavier_normal(RNNG.variable(torch.FloatTensor(*size), gpu))

    @staticmethod
    def variable(tensor, gpu):
        if gpu:
            return torch.autograd.Variable(tensor).cuda()
        else:
            return torch.autograd.Variable(tensor)

    def forward(self,inp):


        lstm_initial = (RNNG.xavier_init(self.use_gpu, 1, self.lstm_hid_dim), RNNG.xavier_init(self.use_gpu, 1, self.lstm_hid_dim))
        empty_emb = nn.Parameter(torch.randn(1, self.lstm_hid_dim))
        self.stack.initialize_tensors(lstm_initial,empty_emb)
        
        self.stack.clear()
        seq_len = inp.shape[0]
        inp = inp.long().view(1,inp.shape[0])
        inp = self.tok_emb(inp)
        inp = inp.transpose(0,1)


        out = []
        for n in range(seq_len):
            i = inp[n:n+1,:,:]
            i = i.view(i.shape[1],i.shape[2])
            #hx,cx = self.lstm_cell(i,(hx,cx))
            self.stack.push(i)
            out.append(self.stack.embedding())

        out = torch.cat(out)
        out = self.linear(out)





        
if __name__=="__main__":


    dummy_input = torch.FloatTensor([1,2,3,0])
    dummy_out = torch.FloatTensor([1])
    print('in dims')
    print(dummy_input.shape)
    tok_embedding_dim = 300
    hidden_dim = 256

    parser = RNNG(dropout_ratio=0.2,tok_emb_dim=300,lstm_hid_dim=256,batch_size=1,output_dim=2,vocab_size=4)
    parser(dummy_input)



