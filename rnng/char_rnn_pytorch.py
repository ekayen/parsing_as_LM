import sys
from lexicons import *
from torch import nn
import torch

class CharRNN(nn.Module):

    """
    That's a basic char rnn, that can be embedded into a larger model.
    This is a bi-lstm kind of model made to be used as a sub-model similarly to other dynet builders.
    """
    def __init__(self,char_embedding_size,hidden_size,charset):
        """
        Creates a CharRNN from scratch. Should be used as an external constructor.
        Args:
            char_embedding_size          (int): size of char embeddings
            bidden_size                  (int): size of RNN hidden
            charset            (SymbolLexicon): a char indexing object
            model  (dynet.ParameterCollection): parameters from the caller model
        """
        super().__init__()
                
        self.char_embedding_size = char_embedding_size
        self.hidden_size         = hidden_size
        self.charset             = charset

        self.emb = nn.Embedding(self.charset.size(),self.char_embedding_size)
        self.rnn = nn.LSTM(num_layers=2,input_size=self.char_embedding_size,hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size,self.hidden_size)
        self.relu = nn.ReLU()
        

    def forward(self,token,device):

        token = list(token)
        char_idx = torch.FloatTensor([[self.charset.index(c)] for c in token if c in self.charset]).long().to(device)
        char_emb = self.emb(char_idx) # time,batch,emb

        """
        if not char_embeddings: #empty word, no char recognized
            print('problematic token',token,file=sys.stderr,flush=True)
            return self.b
        """
        
        hidden = self.init_hidden(device)
        if len(char_emb.shape) < 3:
            import pdb;pdb.set_trace()

        rnn_out,hidden = self.rnn(char_emb,hidden)
        final_out = rnn_out[-1,:,:]
        lin_out = self.fc(final_out)
        out = self.relu(lin_out)
        return out


    def init_hidden(self,device,batch_size=1):
        num_layers = 1
        directions = 2
        h0 = nn.init.xavier_normal_(torch.FloatTensor(num_layers*directions, batch_size, self.hidden_size)).to(device)
        c0 = nn.init.xavier_normal_(torch.FloatTensor(num_layers*directions, batch_size, self.hidden_size)).to(device)
        return (h0,c0)
    
if __name__=="__main__":


    char_emb_size = 100
    hid_size = 256
    
    charset          = set([ ])
    charset.update(list('abcdefghijklmnopqrstuvwxyz'))
    charset = SymbolLexicon(charset)
    rnn = CharRNN(char_embedding_size=char_emb_size,hidden_size=hid_size,charset=charset)

    output = rnn('cat')

               
