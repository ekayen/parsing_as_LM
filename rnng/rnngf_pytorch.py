""" 
This class implements an RNNG parser with class factored word emissions.
"""
import sys
import numpy as np
import numpy.random as npr
import pandas as pda
import getopt
import json
from torch import nn
import torch
import time

from random        import shuffle
from math          import exp,log,floor,isnan
from constree      import *
from lexicons      import *
from proc_monitors import *
from rnng_params   import *
from char_rnn_pytorch import *
from stack_lstm    import *
from utils         import * 

class StackSymbol:
    """
    A convenience class for stack symbols
    """
    PREDICTED = 1
    COMPLETED = 0
    
    def __init__(self,symbol,status,embedding):
        """
        Args:
             symbol           (string): a non terminal or a word
             status             (enum): predicted or completed
             embedding (dy.expression): a dynet expression being the embedding of the subtree dominated by this symbol (or word)
        """
        self.symbol,self.status,self.embedding = symbol,status,embedding

    def copy(self):
        return StackSymbol(self.symbol,self.status,self.embedding)

    def complete(self):
        c = self.copy()
        c.status = StackSymbol.COMPLETED
        return c
    
    def __str__(self):
        s =  '*%s'%(self.symbol,) if self.status == StackSymbol.PREDICTED else '%s*'%(self.symbol,)
        return s


def config2str(configuration):
    #pretty prints a config
    S,B,n,stack_state,lab_state = configuration
 
    stack  = ','.join([str(elt) for elt in S])
    bfr    = ','.join([str(elt) for elt in B])
    return '< (%s) , (%s) , %d>'%(stack,bfr,n)

    
class BeamElement:
    """
    This class is a place holder for elements in the beam.
    """

    __slots__ = ['trueK','prev_element', 'prev_action','prefix_gprob','prefix_dprob','configuration','K','succ_recorded','fail_recorded','overall_recorded']
    
    def __init__(self,prev_element,prev_action,prefix_gprob,prefix_dprob):
        """
        Args:
             prev_element (BeamElement) : the previous element or None
             prev_action       (string) : the action generating this element or None
             prefix_gprob       (float) : prefix generative probability
             prefix_dprob       (float) : prefix discriminative probability
        """
        self.prev_element = prev_element
        self.prev_action  = prev_action
        self.prefix_gprob = prefix_gprob
        self.prefix_dprob = prefix_dprob
        self.configuration = None

        #stats backtrack flags
        self.succ_recorded     = False
        self.fail_recorded     = False
        self.overall_recorded  = False
        
    @staticmethod
    def init_element(configuration):
        """
        Generates the beam initial (root) element
        Args:
           configuration (tuple): the parser init config
        Returns:
           BeamElement to be used at init
        """
        b =  BeamElement(None,None,0,0)
        b.configuration       = configuration
        b.succ_recorded       = True
        b.fail_recorded       = True
        b.overall_recorded    = True
        return b
    
    def is_initial_element(self):
        """
        Returns:
            bool. True if the element is root of the beam
        """
        return self.prev_element is None or self.prev_action is None

    
class RNNGparser(nn.Module):
    """
    This is an RNNG parser with in-order tree traversal
    """
    #action codes
    SHIFT           = '<S>'
    OPEN            = '<O>'
    CLOSE           = '<C>'
    TERMINATE       = '<T>'
    
    #labelling states
    WORD_LABEL      = '@w'
    NT_LABEL        = '@o'
    NO_LABEL        = '@-'
    
    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'

    def __init__(self,brown_clusters,train_treebank,dev_treebank,
                      vocab_thresh=1,\
                      stack_memory_size=100,
                      word_embedding_size=100,
                      char_embedding_size=50,
                      char_memory_size=50):
        """
        Args:
           brown_clusters       (str)  : a filename where to find brown clusters     
        Kwargs:
           vocab_thresh         (int)  : max number of words in the lexical vocab
           stack_embedding_size (int)  : size of stack lstm input 
           stack_memory_size    (int)  : size of the stack and tree lstm hidden layers
           word_embedding_size  (int)  : size of word embeddings
           char_embedding_size  (int)  : size of char lstm input 
           char_memory_size     (int)  : size of char lstm hidden layer
        """
        super(RNNGparser,self).__init__()
        
        
        self.brown_file           = brown_clusters
        self.vocab_thresh         = vocab_thresh
        self.stack_embedding_size = char_embedding_size + word_embedding_size
        self.stack_hidden_size    = stack_memory_size
        self.word_embedding_size  = word_embedding_size
        self.char_embedding_size  = char_embedding_size
        self.char_memory_size     = char_memory_size 
        self.dropout              = 0.0

        #def allocate_structure(self):
        #    """
        #    Allocates memory for the model parameters.
        #    """
        #    self.model                     = dy.ParameterCollection()

        self.code_lexicon(train_treebank)
        self.code_nonterminals(train_treebank,dev_treebank)
        self.code_struct_actions()
    
        #input
        self.nonterminals_embeddings   = nn.Embedding(self.nonterminals.size(),self.stack_embedding_size)
        self.word_embeddings           = nn.Embedding(self.lexicon.size(),self.word_embedding_size)

        #output
        #self.structural_W             = self.model.add_parameters((self.actions.size(),self.stack_hidden_size)) # I think these are covered by an "init hidden" kind of thing
        #self.structural_b             = self.model.add_parameters((self.actions.size()))

        #self.word_softmax             = dy.ClassFactoredSoftmaxBuilder(self.stack_hidden_size,self.brown_file,self.lexicon.words2i,self.model,bias=True)
        self.word_softmax             = nn.LogSoftmax(dim=1) # The above is hierarchical softmax; I will try an implementation of that later.
        self.nt_softmax               = nn.LogSoftmax(dim=1)
        self.structural_softmax       = nn.LogSoftmax(dim=1) # Pretty sure that having separate softmaxes is unnecessary, but I'm not 100% sure how graphs work, so keeping this for now.

        
        #self.nonterminals_W          = self.model.add_parameters((self.nonterminals.size(),self.stack_hidden_size)) # should be handled with an init_hidden fn
        #self.nonterminals_b          = self.model.add_parameters((self.nonterminals.size()))

        #stack_lstm
        
        self.lstm_cell                 = nn.LSTMCell(input_size=self.stack_embedding_size, hidden_size=self.stack_hidden_size)
        self.lstm_dropout              = nn.Dropout(p=self.dropout)
        self.rnn                       = StackRNN(self.lstm_cell,
                                                  self.lstm_dropout,
                                                  self._rnn_get_output)
        self.stack_relu                = nn.ReLU()
        self.word_fc                   = nn.Linear(self.stack_hidden_size,self.lexicon.size())
        self.nt_fc                     = nn.Linear(self.stack_hidden_size,self.nonterminals.size())
        self.structural_fc             = nn.Linear(self.stack_hidden_size,self.actions.size())
        
        #tree bi-lstm
        #self.tree_fwd                 = dy.VanillaLSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)        
        #self.tree_bwd                 = dy.VanillaLSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)

        
        self.tree_fwd                 = nn.LSTMCell(self.stack_embedding_size, self.stack_hidden_size)        
        self.tree_bwd                 = nn.LSTMCell(self.stack_embedding_size, self.stack_hidden_size)


        self.tree_fc                  = nn.Linear(self.stack_hidden_size*2,self.stack_embedding_size)
        self.relu                     = nn.ReLU()
        
        #self.tree_W                   = self.model.add_parameters((self.stack_embedding_size,self.stack_hidden_size*2))
        #self.tree_b                   = self.model.add_parameters((self.stack_embedding_size))
        # EKN is there an equivalent to declaring parameters like this for pytorch?
        # EKN or is this under the hood? or an init function
        
        self.char_rnn                 = CharRNN(self.char_embedding_size,self.char_memory_size,self.charset)

    def _rnn_get_output(self,state):
        return state[0]
        
    @staticmethod
    def xavier_init(device,*size):
        gpu = False
        return nn.init.xavier_normal_(torch.FloatTensor(*size)).to(device)
        
    def code_lexicon(self,treebank):
        """
        Codes a lexicon on integers indexes and generates a lexicon object.
        
        Args:
             treebank       (list) : a list of trees where to extract the words from
             vocab_thresh   (int)  : the count threshold above which the vocabulary is known to the parser
        Returns:
             SymbolLexicon. The bijective encoding
        """
        
        known_vocabulary = []
        charset          = set([ ])
        charset.update(list(RNNGparser.START_TOKEN)) # EKN
        charset.update(['&'])
        for tree in treebank:
            tokens = tree.tokens()
            for word in tokens:
                charset.update(list(word))
            known_vocabulary.extend(tokens) 
        known_vocabulary = get_known_vocabulary(known_vocabulary,vocab_threshold=1)
        known_vocabulary.add(RNNGparser.START_TOKEN)
        self.brown_file  = normalize_brown_file(self.brown_file,known_vocabulary,self.brown_file+'.unk',UNK_SYMBOL=RNNGparser.UNKNOWN_TOKEN)
        self.lexicon     = SymbolLexicon( list(known_vocabulary),unk_word=RNNGparser.UNKNOWN_TOKEN)
        self.charset     = SymbolLexicon(list(charset))
        return self.lexicon

    
    def code_nonterminals(self,train_treebank,dev_treebank):
        """
        Extracts the nonterminals from a treebank and codes them on integers as a lexicon object.
        
        Args:
           train_treebank   (list) : a list of trees  where to extract the non terminals from
           dev_treebank   (list) : a list of trees  where to extract the non terminals from

        Returns:
           SymbolLexicon. The bijective encoding
        """
        nonterminals = set([])
        for tree in train_treebank:
            nonterminals.update(tree.collect_nonterminals())
        for tree in dev_treebank:
            nonterminals.update(tree.collect_nonterminals())
        self.nonterminals = SymbolLexicon(list(nonterminals))
        return self.nonterminals

    
    def code_struct_actions(self):
        """
        Codes the structural actions on integers and generates bool masks

        Returns:
            SymbolLexicon. The bijective encoding
        """
        self.actions         = SymbolLexicon([RNNGparser.SHIFT,RNNGparser.OPEN,RNNGparser.CLOSE,RNNGparser.TERMINATE])

        #Allocates masks
        self.open_mask       = np.array([True]*4)
        self.shift_mask      = np.array([True]*4)
        self.close_mask      = np.array([True]*4)
        self.terminate_mask  = np.array([True]*4)

        self.open_mask[self.actions.index(RNNGparser.OPEN)]           = False
        self.shift_mask[self.actions.index(RNNGparser.SHIFT)]         = False
        self.close_mask[self.actions.index(RNNGparser.CLOSE)]         = False
        self.terminate_mask[self.actions.index(RNNGparser.TERMINATE)] = False

        return self.actions

    @staticmethod
    def load_model(model_name,model_path=None): 
        """
        Loads an RNNG parser from params at prefix model_name

        Args:
            model_name   (string): the prefix path for param files

        Returns:
            RNNGparser. An instance of RNNG ready to use.
        """
        if model_path:
            model_path = os.path.join(model_path,model_name+'.pt')
        else:
            model_path = model_name+'.pt'
            
        checkpoint = torch.load(model_path)
        params = checkpoint['parser_params']
        brown_file = params['brown_file'] 
        train_treebank = params['train']
        dev_treebank = params['dev']
        stack_size = params['stack_size']
        word_emb = params['word_emb']        

        lr = checkpoint['lr']
        
        parser = RNNGparser(brown_file,train_treebank,dev_treebank,stack_size,word_emb)
        parser.load_state_dict(checkpoint['model_state_dict'])
        
        trainer = torch.optim.Adam(parser.parameters(),lr=lr)
        trainer.load_state_dict(checkpoint['opt_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        parser.lexicon = checkpoint['lexicon']
        parser.nonterminals = checkpoint['nonterminals']
        parser.charset = checkpoint['charset']
        #parser.lexicon      = SymbolLexicon.load(model_name+'.lex') 
        #parser.nonterminals = SymbolLexicon.load(model_name+'.nt')  
        #parser.charset      = SymbolLexicon.load(model_name+'.char')
        parser.code_struct_actions()
        return parser,trainer

    def save_model(self,model_name,trainer,epoch,loss,config,lr,model_path=None): 
        """
        Saves the model params using the prefix model_name.

        Args:
            model_name   (string): the prefix path for param files
        """
        """
        hyperparams = { 'brown_file':self.brown_file,\
                        'vocab_thresh':self.vocab_thresh,\
                        'stack_hidden_size':self.stack_hidden_size,\
                        'word_embedding_size':self.word_embedding_size,\
                        'char_memory_size':self.char_memory_size,\
                        'char_embedding_size':self.char_embedding_size}
  
        jfile = open(model_name+'.json','w')
        jfile.write(json.dumps(hyperparams))
        jfile.close()
        """
        print('Saving model ...')
        if model_path:
            save_path = os.path.join(model_path, model_name + '.pt')
        else:
            save_path = os.path.join(model_name + '.pt')
        torch.save({'charset':self.charset,
                    'lexicon':self.lexicon,
                    'nonterminals':self.nonterminals,
                    'epoch':epoch,
                    'parser_params':parser_params,
                    'loss':loss,
                    'model_state_dict':self.state_dict(),
                    'opt_state_dict':trainer.state_dict(),
                    'lr':lr}, save_path)
        print('done')
        
    
    #TRANSITION SYSTEM AND ORACLE
    
    def init_configuration(self,N,device): # DIMS CHECKED
        """
        A configuration is a 5-tuple (S,B,n,stack_mem,labelling?) where:
        
          S: is the stack
          B: the buffer
          n: the number of predicted constituents in the stack
          stack_mem: the current state of the stack lstm
          lab_state: the labelling state of the configuration
        
        This method creates an initial configuration, with empty stack, full buffer (as a list of integers and null score)
        Stacks are filled with non terminals of type StackSymbol.
        Buffers are only filled with terminals of type integer.

        Arguments:
           N  (int) : the length of the input sentence
        Returns:
           tuple. an initial configuration
        """
        word_tens = self.word_embeddings(torch.FloatTensor([self.lexicon.index(RNNGparser.START_TOKEN)]).long().to(device))
        char_tens = self.char_rnn(RNNGparser.START_TOKEN,device)
        e = torch.cat([word_tens,char_tens],dim=1)
        """
        print('INIT CONFIG')
        print(f'e: {e.shape}')
        import pdb;pdb.set_trace()
        """
        self.rnn.clear()
        self.rnn.push(e) # UNK is there some way to do the equivalent of passing the stack state?
        # Or do I just need to push/pop from the stack, rather than store the top of the stack (which contains a pointer)
        return ([],tuple(range(N)),0,self.rnn,RNNGparser.NO_LABEL)
    
    def shift_action(self,configuration):
        """
        This performs a shift action.
        That is the parser commits itself to generate a word at the next step.

        Arguments:
           configuration (tuple) : a configuration frow where to shift
        Returns:
           tuple. a configuration resulting from shift 
        """
        S,B,n,stack_state,lab_state = configuration
        return (S,B,n,stack_state,RNNGparser.WORD_LABEL)

    def generate_word(self,configuration,sentence,device):
        """
        This generates a word (performs the actual shifting).
        
        Arguments:
            configuration (tuple) :  a configuration frow where to generate a word
            sentence      (list)  :  a list of strings, the sentence tokens
        Returns:
           tuple. a configuration after word generation
        """
        S,B,n,stack_state,lab_state = configuration
        word_emb = self.word_embeddings(torch.FloatTensor([self.lexicon.index(sentence[B[0]])]).long().to(device))
        char_emb = self.char_rnn(sentence[B[0]],device)
        e = torch.cat([word_emb, char_emb],dim=1)
        """
        print('GENERATE WORD')
        print(f'word_emb: {word_emb.shape}')
        print(f'char_emb: {char_emb.shape}')
        print(f'e: {e.shape}')
        import pdb;pdb.set_trace()
        """
        self.rnn.push(e)
        return (S + [StackSymbol(B[0],StackSymbol.COMPLETED,e)],B[1:],n,self.rnn,RNNGparser.NO_LABEL)

    def open_action(self,configuration):
        """
        The Open action commits the parser to open a constituent without doing it immediately
        Arguments:
           configuration (tuple): a configuration frow where to perform open
        Returns:
           tuple. A configuration resulting from opening the constituent
        """
        S,B,n,stack_state,lab_state = configuration 
        #EKNprint('open action -- no input to stack state')
        return (S,B,n,stack_state,RNNGparser.NT_LABEL)

    def close_action(self,configuration,device): # DIMS CHECKED
        """
        This actually executes the RNNG CLOSE action.
        The close action also commits the parser to score the closed constituent label at next step
        Arguments:
           configuration (tuple): a configuration frow where to perform open
        Returns:
           tuple. A configuration resulting from closing the constituent
        """
        S,B,n,stack_state,lab_state = configuration

        assert(n > 0)
        
        newS = S[:]
        closed_symbols = []
        while newS[-1].status != StackSymbol.PREDICTED:
            closed_symbols.append(newS.pop())
            self.rnn.pop() # UNK make sure this works and actually makes the rnn point to the prev state
        self.rnn.pop()
        
        # tree rnn
        # TODO this bit should be a single call to LSTM (not LSTM cell),
        # but I'm weirded out by the fact both
        # fwd and bwd start with the same symbol.
        # LSTM() wouldn't allow that, so I'm doing it by hand with LSTMCell for now till I can figure out if it matters.

        nt_emb = torch.FloatTensor([self.nonterminals.index(newS[-1].symbol)]).long().to(device)
        nonterm_emb = self.nonterminals_embeddings(nt_emb)

        tree_init = (RNNGparser.xavier_init(device,1,self.stack_hidden_size),RNNGparser.xavier_init(device,1,self.stack_hidden_size))
        
        fwd_h,fwd_c = self.tree_fwd(nonterm_emb,tree_init) 
        for SYM in reversed(closed_symbols):
            fwd_h,fwd_c = self.tree_fwd(SYM.embedding,(fwd_h,fwd_c))
            
        bwd_h,bwd_c = self.tree_bwd(nonterm_emb,tree_init)
        for SYM in closed_symbols:
            bwd_h,bwd_c = self.tree_bwd(SYM.embedding,(bwd_h,bwd_c))

        tree_h         = torch.cat([self.lstm_dropout(fwd_h),self.lstm_dropout(bwd_h)],dim=1)
        tree_embedding = self.relu(self.tree_fc(tree_h))
        """
        print('CLOSE ACTION')
        print('tree h')
        print(tree_h.shape)
        print('tree fc')
        print(self.tree_fc)
        print('tree_emb')
        print(tree_embedding.shape)
        """
        newS[-1] = newS[-1].complete() 
        newS[-1].embedding = tree_embedding
        self.rnn.push(tree_embedding)
        return (newS,B,n-1,self.rnn,RNNGparser.NO_LABEL)


    def open_nonterminal(self,configuration,Xlabel,device):
        """
        The nonterminal labelling action. This adds an open nonterminal on the stack under the stack top (left corner style inference)
        
        Arguments:
            configuration (tuple) : a configuration where to perform the labelling
            Xlabel        (string): the nonterminal label
        Returns:
            tuple. A configuration resulting from the labelling
        """
        S,B,n,stack_state,lab_state = configuration
        
        stack_top = S[-1]
        nt_emb = torch.FloatTensor([self.nonterminals.index(Xlabel)]).long().to(device)
        e = self.nonterminals_embeddings(nt_emb)
        """
        print('OPEN NONTERMINAL')
        print(f'e: {e.shape}')
        print(f'nt_emb: {nt_emb}')
        print(f'stack_top: {stack_top}')
        import pdb;pdb.set_trace()
        """
        self.rnn.push(e)
        return (S[:-1] + [StackSymbol(Xlabel,StackSymbol.PREDICTED,e),stack_top],B,n+1,self.rnn,RNNGparser.NO_LABEL)

    def close_nonterminal(self,configuration,Xlabel):
        """
        This is essentially a No Op.

        Arguments:
            configuration (tuple) : a configuration where to perform the closure
        Returns:
            tuple. A configuration resulting from the closure.
        """
        S,B,n,stack_state,lab_state = configuration
        return (S,B,n,stack_state,RNNGparser.NO_LABEL)
    
    def static_inorder_oracle(self,ref_tree,sentence,device,configuration=None):
        """
        This generates a simple oracle derivation by performing an inorder traversal of the reference tree.
        The function simulates parsing and thus checks for oracle soundness and also generates reference configurations.

        Arguments:
            ref_tree (ConsTree)   : a local tree root
            sentence (list)       : a list of strings, the tokens of the sentence.
        Kwargs:
            configuration (tuple) : the current configuration
        Returns:
            (a list of actions,the resulting configuration). Actions in the derivation are coded as strings 
        """
        is_root = False
        if configuration is None:
            configuration = self.init_configuration(len(sentence),device)
            is_root = True
            
        if ref_tree.is_leaf():
            if not self.actions.index(RNNGparser.SHIFT) in self.allowed_structural_actions(configuration):
                print("oracle unsound <shift> ",configuration,ref_tree)
            configuration = self.shift_action(configuration)
            configuration = self.generate_word(configuration,sentence,device)
            return ( [RNNGparser.SHIFT, ref_tree.label], configuration)
        
        else:
            first_child = ref_tree.children[0]
            derivation, configuration = self.static_inorder_oracle(first_child,sentence,device,configuration)

            if not self.actions.index(RNNGparser.OPEN) in self.allowed_structural_actions(configuration):
                print('oracle unsound <open>',ref_tree)
            configuration = self.open_action(configuration)
            configuration = self.open_nonterminal(configuration,ref_tree.label,device)
            derivation.extend([RNNGparser.OPEN,ref_tree.label])
            
            for child in ref_tree.children[1:]:
                subderivation,configuration = self.static_inorder_oracle(child,sentence,device,configuration) 
                derivation.extend(subderivation)
                
            if not self.actions.index(RNNGparser.CLOSE) in self.allowed_structural_actions(configuration):
                print('oracle unsound <close>',ref_tree)
            configuration = self.close_action(configuration,device)
            derivation.append(RNNGparser.CLOSE)
            
        if is_root:
             derivation.append(RNNGparser.TERMINATE)
             
        return (derivation,configuration)
     
    def allowed_structural_actions(self,configuration,disallowed=False):
        """
        Returns the list of structural actions allowed given this configuration.
        Arguments:
           configuration          (tuple) : a configuration
        Returns:
           a list. Indexes of the allowed actions
        """        
        S,B,n,stack_state,lab_state = configuration 
        MASK = np.array([True] * self.actions.size())
        
        if not S or (len(S) >= 2 and S[-2].status == StackSymbol.PREDICTED):
            #last condition prevents unaries and takes into account the reordering of open
            MASK *= self.open_mask
        if B or n != 0 or len(S) > 1:
            MASK *= self.terminate_mask
        if not B or (S and n == 0):
            MASK *= self.shift_mask
        if not S or n < 1 or (len(S) >=2 and S[-2].status == StackSymbol.PREDICTED and S[-1].symbol in self.nonterminals):
            # last condition prevents unaries and takes into account the reordering of open;
            # exceptional unaries are allowed on top of terminals symbols only
                MASK *= self.close_mask

        allowed_idxes = [idx for idx, mask_val in enumerate(MASK) if mask_val]
        if disallowed:
            disallowed_idxes = [idx for idx,mask_val in enumerate(MASK) if not mask_val]
            return disallowed_idxes
        else:
            return allowed_idxes


        
    def ifdropout(self,expression):
        """
        Applies dropout to a dynet expression only if dropout > 0.0.
        """
        return self.lstm_dropout(expression,self.dropout) if self.dropout > 0.0 else expression

    
    def predict_action_distrib(self,configuration,sentence):
        """
        Predicts the log distribution for next actions from the current configuration.

        Args:
          configuration           (tuple): the current configuration
          sentence                 (list): a list of string, the tokens

        Returns:
            a list of couples (action, log probability). The list is empty if the parser is trapped (aka no action is possible).
            currently returns a zip generator.
        """
        
        S,B,n,stack_state,lab_state = configuration

        if lab_state == RNNGparser.WORD_LABEL:
            next_word     = (sentence[B[0]])
            next_word_idx = self.lexicon.index(next_word)
            #return [(next_word,-self.word_softmax.neg_log_softmax(dy.rectify(stack_state.output()),next_word_idx).value())] 
            return [(next_word,self.word_softmax(self.word_fc(self.stack_relu(self.rnn.embedding())),next_word_idx))] # TODO replace with hierarchical softmax eventually
        elif lab_state == RNNGparser.NT_LABEL :
            #logprobs = dy.log_softmax(self.nonterminals_W  * dy.rectify(stack_state.output())  + self.nonterminals_b).value() #removed rectifier
            logprobs = self.nt_softmax(self.nt_fc(self.stack_relu(self.rnn.embedding())))
            return zip(self.nonterminals.i2words,logprobs)
        elif lab_state == RNNGparser.NO_LABEL :
            restr = self.allowed_structural_actions(configuration)
            if restr:
                print(f'restr: {restr}')
                logprobs =  self.structural_softmax(mask_by_idx(self.structural_fc(self.relu(self.rnn.embedding())),restr,dim=1)) # TODO check dim
                return [ (self.actions.wordform(action_idx),logprob) for action_idx,logprob in zip(range(self.actions.size()),logprobs) if action_idx in restr]
        #parser trapped...
        return []


    def eval_action_distrib(self,configuration,sentence,ref_action):
        """
        Evaluates the model predictions against the reference data.

        Args:
          configuration   (tuple): the current configuration
          sentence         (list): a list of string, the tokens
          ref_action     (string): the reference action.
          
        Returns:
            a dynet expression. The loss (NLL) for this action
        """
        criterion = nn.CrossEntropyLoss()
        S,B,n,stack_state,lab_state = configuration
        if lab_state == RNNGparser.WORD_LABEL:
            ref_idx  = torch.FloatTensor([self.lexicon.index(ref_action)]).long().to(device)
            #stack_output = self.stack_relu(self.rnn.embedding())
            #dropped_out = self.lstm_dropout(stack_output)
            #fc_out = self.word_fc(dropped_out)
            #nll = criterion(fc_out,ref_idx)
           
            stack_output = self.word_fc(self.lstm_dropout(self.stack_relu(self.rnn.embedding())))
            nll = criterion(stack_output,ref_idx)
            #soft_out = self.word_softmax(fc_out)
            #nll = -soft_out[:,ref_idx]
            #nll =  self.softmax(self.lstm_dropout(stack_output))[ref_idx]
        elif lab_state == RNNGparser.NT_LABEL:
            ref_idx  = torch.FloatTensor([self.nonterminals.index(ref_action)]).long().to(device)

            #stack_output = self.stack_relu(self.rnn.embedding())
            #dropped_out = self.lstm_dropout(stack_output)
            #fc_out = self.nt_fc(dropped_out)
            #soft_out = self.nt_softmax(fc_out)
            #nll = -soft_out[:,ref_idx] # TODO check that my strategy to pick the right one worked
            #nll = criterion(fc_out,ref_idx)

            stack_output = self.nt_fc(self.lstm_dropout(self.stack_relu(self.rnn.embedding())))
            nll = criterion(stack_output,ref_idx)

        elif lab_state == RNNGparser.NO_LABEL:
            ref_idx = torch.FloatTensor([self.actions.index(ref_action)]).long().to(device)
            restr = self.allowed_structural_actions(configuration)
            stack_output = self.structural_fc(self.lstm_dropout(self.stack_relu(self.rnn.embedding())))
            masked_fc_out = mask_by_idx(stack_output,restr,dim=1)
            #soft_out = self.structural_softmax(masked_fc_out)
            #neg_log_soft = -soft_out
            #long_nll = neg_log_soft[:,ref_idx] # TODO check mask AND pick and dim
            nll = criterion(masked_fc_out,ref_idx)
            if nll.isnan(): import pdb;pdb.set_trace()
        else:
            print('error in evaluation')

        return nll

    def eval_sentences(self,ref_tree_list,device,trainer=None,backprop=True):
        """
        Evaluates the model predictions against the reference data.
        and optionally performs backpropagation. 

        The function either takes a single tree or a batch of trees (as list) for evaluation.
        
        Args:
          ref_tree_list    (ConsTree) or (list): a list of reference tree or a single tree.
        Kwargs:
          backprop                       (bool): a flag telling if we perform backprop
        Returns:
          RuntimeStats. the model NLL, the word only NLL, the size of the derivations, the number of predicted words on this batch
        """
        lstm_initial = (RNNGparser.xavier_init(device,1, self.stack_hidden_size), RNNGparser.xavier_init(device,1, self.stack_hidden_size))
        empty_emb = nn.Parameter(torch.randn(1, self.stack_embedding_size)).to(device)
        self.rnn.initialize_tensors(lstm_initial,empty_emb)
        
        
        sent_num = 0
        if not backprop:
            parser.eval()
        else:
            parser.train()
            trainer.zero_grad()
        #dropout = self.dropout
        #if not backprop:
        #    self.dropout = 0.0
        
        ref_trees = [ref_tree_list] if type(ref_tree_list) != list else ref_tree_list

        loss = 0
        lex_loss = 0
    
        runstats = RuntimeStats('NLL','lexNLL','N','lexN')
        runstats.push_row()
        
        for ref_tree in ref_trees:

            sentence = ref_tree.tokens()

            sent_num += 1

            derivation,last_config = self.static_inorder_oracle(ref_tree,sentence,device,configuration=None)
            
            runstats['lexN']  += len(sentence)
            runstats['N']  += len(derivation)
    
            configuration = self.init_configuration(len(sentence),device)
            
            for i,ref_action in enumerate(derivation):
                S,B,n,stack_state,lab_state = configuration
                nll =  self.eval_action_distrib(configuration,sentence,ref_action)
                if nll.isnan(): import pdb;pdb.set_trace()
                loss += nll
                #all_NLL.append( nll )
                if lab_state == RNNGparser.WORD_LABEL:
                    configuration = self.generate_word(configuration,sentence,device)
                    #lexical_NLL.append(nll)
                    lex_loss += nll#.data[0]
                elif lab_state == RNNGparser.NT_LABEL:
                    configuration = self.open_nonterminal(configuration,ref_action,device)
                elif ref_action == RNNGparser.CLOSE:
                    configuration = self.close_action(configuration,device)
                elif ref_action == RNNGparser.OPEN:
                    configuration = self.open_action(configuration)
                elif ref_action == RNNGparser.SHIFT:
                    configuration = self.shift_action(configuration)
                elif ref_action == RNNGparser.TERMINATE:
                    pass
                

        
        runstats['NLL']   = loss.detach() # TODO maybe remove grad?
        runstats['lexNLL'] = lex_loss.detach()
        if backprop:
            print(loss)
            loss.backward()
            """
            for nm, param in self.named_parameters():
                if param.requires_grad:
                    print(nm, param.grad)
                    import pdb;pdb.set_trace()
            """
            try:
                trainer.step()
                trainer.zero_grad()
            except RuntimeError:
                print('\nGradient exploded, batch update aborted...')
        #else:
        #    self.dropout = dropout
        return runstats

    def summary(self,train_bank_size,dev_bank_size,learning_rate,batch_size,epochs):
        """
        A summary to display before training. Provides model structure and main learning hyperparams

        Args:
            train_bank_size  (int): num training trees
            dev_bank_size    (int): num dev trees
            learning_rate  (float): the learning rate
            batch_size       (int): size of minibatch
            epochs           (int): num epochs
        Returns:
            string. The summary
        """
        return '\n'.join(['----------------------------',\
                          'Vocabulary   size   : %d'%(self.lexicon.size()),\
                          '# Nonterminals      : %d'%(self.nonterminals.size()),\
                          'Word embedding size : %d'%(self.word_embedding_size),\
                          'Stack embedding size: %d'%(self.stack_embedding_size),\
                          'Stack memory size   : %d'%(self.stack_hidden_size),\
                          'Char embedding size : %d'%(self.char_embedding_size),\
                          'Char memory size    : %d'%(self.char_memory_size),\
                          '',\
                          '# training trees    : %d'%(train_bank_size),\
                          '# validation trees  : %d'%(dev_bank_size),\
                          '# epochs            : %d'%(epochs),\
                          'Learning rate       : %.3f'%(learning_rate),\
                          'Batch size          : %d'%(batch_size),\
                          'Dropout             : %.3f'%(self.dropout),\
                          '----------------------------']) 

    ###  PARSING & SEARCH  #################################################
    def exec_beam_action(self,beam_elt,sentence):
        """
        Generates the element's configuration and assigns it internally.

        Args:
             beam_elt  (BeamElement): a BeamElement missing its configuration
             sentence         (list): a list of strings, the tokens.
        """
        
        if  beam_elt.is_initial_element(): 
            beam_elt.configuration = self.init_configuration(len(sentence))
        else:
            configuration = beam_elt.prev_element.configuration
            S,B,n,stack_state,lab_state = configuration
                        
            if lab_state == RNNGparser.WORD_LABEL:
                beam_elt.configuration = self.generate_word(configuration,sentence,device)
            elif lab_state == RNNGparser.NT_LABEL:
                beam_elt.configuration = self.open_nonterminal(configuration,beam_elt.prev_action,device)
            elif beam_elt.prev_action == RNNGparser.CLOSE:
                beam_elt.configuration = self.close_action(configuration,device)
            elif beam_elt.prev_action == RNNGparser.OPEN:
                beam_elt.configuration = self.open_action(configuration)
            elif beam_elt.prev_action == RNNGparser.SHIFT:
                beam_elt.configuration = self.shift_action(configuration)
            elif beam_elt.prev_action == RNNGparser.TERMINATE:
                beam_elt.configuration = configuration
            else:
                print('oops')

    @staticmethod
    def sample_dprob(beam,K):
        """
        Samples without replacement K elements in the beam proportional to their *discriminative* probability
        Inplace destructive operation on the beam.
        Args:
             beam  (list) : a beam data structure
             K      (int) : the number of elts to keep in the Beam
        Returns:
             The beam object
        """
        probs      = np.exp(np.array([elt.prefix_dprob  for elt in beam[-1]])) + np.finfo(float).eps
        probs     /= probs.sum()
        #print(len(beam[-1]),K,probs)
        samp_idxes = npr.choice(list(range(len(beam[-1]))),size=min(len(beam[-1]),K),p=probs,replace=False)
        beam[-1]   = [ beam[-1][idx] for idx in samp_idxes]
        return beam

    @staticmethod
    def prune_dprob(beam,K):
        """
        Prunes the beam to the top K elements using the *discriminative* probability (performs a K-Argmax).
        Inplace destructive operation on the beam.
        Args:
             beam  (list) : a beam data structure
             K       (int): the number of elts to keep in the Beam
        Returns:
             The beam object
        """
        beam[-1].sort(key=lambda x:x.prefix_dprob,reverse=True)
        beam[-1] = beam[-1][:K]
        return beam

    @staticmethod
    def weighted_derivation(success_elt):
        """
        Generates a weighted derivation as a list (Action,logprob)_0 ... (Action,logprob)_m. from a successful beam element
        Args:
            success_elt (BeamElement): a terminated beam element
        Returns:
            list. A derivation is a list of couples (string,float)
        """
        D = []
        current = success_elt
        while not current.is_initial_element():
            D.append((current.prev_action,current.prefix_gprob))
            current = current.prev_element
        D.reverse()
        return D

    @staticmethod
    def beam2stats(success,failures,prefix_prob2,prefix_entropy2):
        """
        Computes statistical indicators of interest from a beam at some timestep t
        Args:
           success         (list): a list of successful BeamElements returned by the parser's search method at time step t
           failures        (list): a list of failures BeamElements returned by the parser's search method at time step t
           prefix_prob2   (float): log probability of the prefix string at timestep t-1
           prefix_entropy2(float): entropy at time step t-1
        Returns:
           A dictionary with collected stats names and their values
        """
        def backtrack_succ(deriv_elt):
            #that's a backtrack with a memo functionality
            c = 0
            while not deriv_elt.succ_recorded:
                c =  c + 1  if deriv_elt.prev_action in [RNNGparser.SHIFT,RNNGparser.OPEN,RNNGparser.CLOSE] else c
                deriv_elt.succ_recorded = True
                deriv_elt = deriv_elt.prev_element
            return c
        
        def backtrack_fail(deriv_elt):
            #that's a backtrack with a memo functionality
            c = 0
            while not deriv_elt.fail_recorded:
                c =  c + 1  if deriv_elt.prev_action in [RNNGparser.SHIFT,RNNGparser.OPEN,RNNGparser.CLOSE] else c
                deriv_elt.fail_recorded = True
                deriv_elt = deriv_elt.prev_element
            return c

        def backtrack_overall(deriv_elt):
            #that's a backtrack with a memo functionality
            c = 0
            while not deriv_elt.overall_recorded:
                c =  c + 1  if deriv_elt.prev_action in [RNNGparser.SHIFT,RNNGparser.OPEN,RNNGparser.CLOSE] else c
                deriv_elt.overall_recorded = True
                deriv_elt = deriv_elt.prev_element
            return c

        #structural metrics
        beam_size          = len(success)
        succ_activity      = sum([ backtrack_succ(elt) for elt in success])
        fail_activity      = sum([ backtrack_fail(elt) for elt in failures])
        overall_activity   = sum([ backtrack_overall(elt) for elt in success+failures])

        #Log likelihood etc...   
        logprobs2              = [elt.prefix_gprob/np.log(2) for elt in success] #change logprobs from base e to base 2
        #logprobs2             = [(elt.prefix_gprob-elt.prefix_dprob+np.log(elt.trueK))/np.log(2) for elt in success] 
        #actualK                = sum(elt.trueK for elt in success)
        #marginal_logprob2      = np.logaddexp2.reduce(logprobs2) - np.log2(actualK)  
        marginal_logprob2      = np.logaddexp2.reduce(logprobs2)
        cond_logprobs2         = [elt-marginal_logprob2 for elt in logprobs2]
 
        #information theoretic metrics 
        entropy_norm           = np.log2(beam_size) if beam_size > 1 else 1
        entropy                = - sum( [np.exp2(logp2)*logp2 for logp2 in cond_logprobs2] ) / entropy_norm
        entropy_reduction      = max(0, prefix_entropy2 - entropy)
        surprisal              = -(marginal_logprob2-prefix_prob2)
        
        return {"beam_size":beam_size,\
                "succ_activity":succ_activity,\
                "fail_activity":fail_activity,\
                "overall_activity":overall_activity,\
                'prefix_logprob':marginal_logprob2,\
                'surprisal':surprisal,\
                'entropy':entropy,\
                'entropy_reduction':entropy_reduction}

    def gather_stats(self,sentence,successes,failures):
        """
        Gathers statistics from the beam
        Args:
           sentence         (list): a list o strings, the input tokens
           successes(list of list): a list of list of beam elements
           failures (list of list): a list of list of beam elements
        Returns:
           (NLL2, pandas DataFrame). a couple with the Negative LoglikeLihood (in base 2 !!!) of the sentence and a Dataframe with word aligned stats aggregated over the list of derivations. 
           The stats collected are such as avg number of OPEN CLOSE since last word, P(w_i| w_i<)...
        """
        N        = len(sentence) 
        #assert(N == len(successes) == len(failures)) 

        prefix_prob  = 0.0
        prefix_entropy = 0.0
        datalines      = [ ]
        nll2           = 0.0
        for idx,token in enumerate(sentence):
        
            stats_dic           = self.beam2stats(successes[idx],failures[idx],prefix_prob,prefix_entropy)
            stats_dic['word']   = token
            stats_dic['is_unk'] = token not in self.lexicon
            datalines.append(stats_dic)
            
            loc_nll2            = -(stats_dic['prefix_logprob']-prefix_prob)   
            nll2               += loc_nll2
            
            prefix_prob         = stats_dic['prefix_logprob']
            prefix_entropy      = stats_dic['entropy']
            
        df = pda.DataFrame(datalines)
        return (-prefix_prob,df) #(nll2,df)

    
    @staticmethod
    def deriv2tree(weighted_derivation):
        """
        Generates a ConsTree from a parse derivation
        Args:
           weighted_derivation (list): a list [ (Action,logprob)_0 ... (Action,logprob)_m ].
        Returns:
           The ConsTree root.
        """
        stack = []  #contains (ConsTree,flag) where flag tells if the constituent is predicted or completed

        prev_action = None
        for action,p in weighted_derivation:
            if prev_action == RNNGparser.SHIFT:
                stack.append( (ConsTree(action),True) )
            elif prev_action == RNNGparser.OPEN:
                lc_child,flag = stack.pop()
                stack.append( (ConsTree(action,children=[lc_child]),False))
            elif action ==  RNNGparser.CLOSE:
                children = []
                while stack:
                    node,completed = stack.pop()
                    if completed:
                        children.append(node)
                    else:
                        for c in reversed(children):
                            node.add_child(c)
                        stack.append((node,True))
                        break
            prev_action = action

        root,flag = stack.pop()
        assert(not stack and flag)
        return root

    def particle_beam_search(self,sentence,device,K=100,alpha=0.0,upper_lex_size=1000):
        """
        Particle filter inspired beam search.
        Args:
              sentence      (list): list of strings (tokens)
        Kwargs:
              K              (int): the number of particles to use
              alpha        (float): the smoothing constant exponentiating the selection step \in [0,1].
                                    the closer to 0,the more uniform the weight distrib, the closer to 1, the more peaked the weight distrib.
              upper_lex_size(int) : an upper bound on the size of the lex beam (in principle never reached) to prevent search explosion in case of adversarial examples.
        Returns:
              (list,list,list). (List of BeamElements, the successes ;  List of list BeamElements local successes ,   List of list BeamElements global failures) 
        """
        #dy.renew_cg()

        init   = BeamElement.init_element(self.init_configuration(len(sentence),device))
        init.K = K
        init.trueK = K
        beam                   = [ init ]
        nextword, nextfailures = [      ],[ ]
        successes              = [ ]

        while beam:
          nextword.append([ ])
          nextfailures.append([ ])
          while beam:                                                               #search step

            elt = beam.pop()
            configuration = elt.configuration
            has_succ     = False

            predictions  = list(self.predict_action_distrib(configuration,sentence))

            if predictions:
              #renormalize here so that it sums to 1 : useful for deficient prob distrib (avoids dropping some particle mass)     
              Z            = np.logaddexp.reduce( [logprob for action,logprob in predictions] )
              for action,logprob in predictions:
                importance_prob  = logprob-Z
                new_K = round(elt.K * exp(importance_prob))
                if new_K > 0.0:
                  new_elt   = BeamElement(elt,action,elt.prefix_gprob+logprob,elt.prefix_dprob+importance_prob)
                  new_elt.K = new_K
                  new_elt.trueK = elt.K * exp(importance_prob)
                  has_succ = True
                  if elt.prev_action == RNNGparser.SHIFT:
                    self.exec_beam_action(new_elt,sentence)     
                    nextword[-1].append(new_elt) 
                  elif action == RNNGparser.TERMINATE: 
                    self.exec_beam_action(new_elt,sentence)    
                    successes.append(new_elt)
                  else:
                    self.exec_beam_action(new_elt,sentence)    
                    beam.append(new_elt)
                
            if not has_succ:
                nextfailures[-1].append(elt)
        
          #select
          weights  = [ elt.K * np.exp(elt.prefix_gprob-elt.prefix_dprob)  for elt in nextword[-1]]
          #weights = [ exp(elt.prefix_gprob) for elt in nextword[-1] ]
          Z       = sum(weights)
          beam.clear()
          if Z > 0:
            weights        = [ w/Z for w in weights ]
            weighted_elts  = [ (elt,weight) for elt,weight in zip(nextword[-1],weights) if round(K * weight) > 0 ] #filtering
            if len(weighted_elts) > upper_lex_size:                                                                #truncates extra large beams
              #print('Beam exploded ! ',len(weighted_elts),file=sys.stdout)
              weighted_elts.sort(key=lambda x: x[1],reverse=True)
              weighted_elts = weighted_elts[:upper_lex_size]

            for elt,weight in weighted_elts:
              elt.K = round(K * weight)
              beam.append(elt)
            
        successes.sort(key=lambda x:x.prefix_gprob,reverse=True)
        return successes,nextword,nextfailures

    
    def predict_beam_generative(self,sentence,K):
        """
        Performs generative parsing and returns an ordered list of successful beam elements.
        This is direct generative parsing. 
        Args:
              sentence      (list): list of strings (tokens)
              K              (int): beam width 
        Returns:
             list. List of BeamElements. 
        """
        Kw  = int(K/10)
        Kft = int(K/100)
        
        #dy.renew_cg()
        init = BeamElement.init_element(self.init_configuration(len(sentence)))
        beam,successes,failures  = [ [init] ],[ ],[ [ ] ]
        
        while beam[-1]:
            
            this_word     = beam[-1]
            next_failures = [ ]
            next_word     = [ ]            
            while this_word and len(next_word) < K:
                    fringe     = [ ]
                    fast_track = [ ]
                    for elt in this_word:
                        configuration = elt.configuration
                        for (action, logprob) in self.predict_action_distrib(configuration,sentence):
                            if elt.prev_action == RNNGparser.SHIFT: #<=> we currently generate a word
                                new_elt = BeamElement(elt,action,elt.prefix_gprob+logprob,elt.prefix_dprob)
                                fast_track.append(new_elt)
                            else:
                                new_elt = BeamElement(elt,action,elt.prefix_gprob+logprob,elt.prefix_dprob+logprob)
                                fringe.append(new_elt)
                                
                    fast_track.sort(key=lambda x:x.prefix_gprob,reverse=True)
                    fast_track = fast_track[:Kft]
                    fringe.sort(key=lambda x:x.prefix_gprob,reverse=True)
                    fringe = fringe[:K-len(fast_track)]+fast_track
                    
                    #here compare this_word and fringe : those elements of
                    #this_word with no successors are failures
                    for elt in this_word:
                        has_succ = False
                        for succ in fringe:
                            if succ.prev_element is elt: #! address itentity
                                    has_succ = True
                                    break
                        if not has_succ:
                            next_failures.append(elt)
                    ####
                    this_word = [ ]
                    for s in fringe:
                        prev_prev_action    = s.prev_element.prev_action
                        if prev_prev_action == RNNGparser.SHIFT: #<=> tests if we currently generate a word
                            next_word.append(s)
                        elif s.prev_action ==  RNNGparser.TERMINATE:
                            successes.append(s)
                        else:
                            self.exec_beam_action(s,sentence)
                            this_word.append(s)
                            
            next_word.sort(key=lambda x:x.prefix_gprob,reverse=True)
            next_failures.extend(next_word[Kw:])
            next_word = next_word[:Kw]
            for elt in next_word:
                self.exec_beam_action(elt,sentence)
            beam.append(next_word)
            failures.append(next_failures)
        if successes:
            successes.sort(key=lambda x:x.prefix_gprob,reverse=True)
            successes = successes[:K]
        return (successes,beam,failures)

    
    def predict_greedy(self,sentence):
        """
        Greedy prediction. Mostly a debug function.
        
        Args: 
              sentence      (list): list of strings (tokens)
        Returns:
              a successful BeamElement if any or None
        """
        #dy.renew_cg()
        current = BeamElement.init_element(self.init_configuration(len(sentence)))
        
        while not current is None:
            configuration               = current.configuration
            S,B,n,stack_state,lab_state = configuration
            next_elt = None
            if lab_state == RNNGparser.WORD_LABEL:
                    for (action, logprob) in self.predict_action_distrib(configuration,sentence):                    
                            next_elt = BeamElement(current,action,current.prefix_gprob+logprob,current.prefix_dprob)
                            self.exec_beam_action(next_elt,sentence)
            elif lab_state == RNNGparser.NT_LABEL:
                    maxval = -np.inf
                    for action,prob in self.predict_action_distrib(configuration,sentence):
                            if prob > maxval:
                                next_elt = BeamElement(current,action,current.prefix_gprob+prob,current.prefix_dprob+prob)
                                maxval = prob
                    self.exec_beam_action(next_elt,sentence)
            else:
                maxval = -np.inf
                for (action, prob) in self.predict_action_distrib(configuration,sentence):
                    if prob > maxval:
                        next_elt = BeamElement(current,action,current.prefix_gprob+prob,current.prefix_dprob+prob)
                        maxval = prob
                     
                if next_elt.prev_action == RNNGparser.TERMINATE:
                    return next_elt
                else:
                    self.exec_beam_action(next_elt,sentence)
            current = next_elt
        return None
          
    def oracle_mode(self,tree,K):
        """
        Parses a sentence in gold oracle mode.
        """
        wordsXtags         = tree.pos_tags()
        tokens             = [tagnode.get_child().label for tagnode in wordsXtags]
        tags               = [tagnode.label for tagnode in wordsXtags]
        #results            =  self.particle_beam_search(tokens,K)
        results            = self.predict_beam_generative(tokens,K)
        fmax,idx_ref = 0,-1
        rmax = None
        for idx,r in enumerate(results):
            r_derivation   = RNNGparser.weighted_derivation(r)
            r_tree         = RNNGparser.deriv2tree(r_derivation)
            r_tree.expand_unaries()
            r_tree.add_gold_tags(tags)
            _,_,F = tree.compare(r_tree)
            if F > fmax:
                fmax    = F
                rmax    = r
                if F == 1.0:
                    idx_ref = idx
                
        print("Best tree")
        r_derivation   = RNNGparser.weighted_derivation(results[0])
        r_tree         = RNNGparser.deriv2tree(r_derivation)
        r_tree.expand_unaries()
        r_tree.add_gold_tags(tags)
        print(r_tree, 'P = ', results[0].prefix_gprob,"L = ",len(r_derivation))
        print()
        print('Ref tree at ',idx_ref)
        print(tree, 'P = ','?' if idx_ref < 0 else results[idx_ref].prefix_gprob, 'L = ', '?' if idx_ref < 0 else len(RNNGparser.weighted_derivation(results[idx_ref])))
        print()
        print('=============================',flush=True)
        return [rmax]
    
    def parse_corpus(self,istream,ostream,device,stats_stream=None,K=10,kbest=1,evalb_mode=False):
        """
        Parses a corpus and prints out the trees in a file.
        Args:
           istream  (stream): the stream where to read the data from
           ostream  (stream): the stream where to write the data to
        Kwargs:
           stats_stream (string): the stream where to dump stats
           K               (int): the size of the beam
           kbest           (int): the number of parses outputted per sentence (<= K)
           evalb_mode     (bool): take an ptb bracketed .mrg file as input and reinserts the pos tags as a post processing step. evalb requires pos tags
        """

        
        self.dropout = 0.0
        NLL2 = 0
        N   = 0
        stats_header = True  
        for line in istream:
                results,words,fails    = None,None,None
                if evalb_mode:
                    tree               = ConsTree.read_tree(line) 
                    wordsXtags         = tree.pos_tags()
                    tokens             = [tagnode.get_child().label for tagnode in wordsXtags]
                    tags               = [tagnode.label for tagnode in wordsXtags]
                    results,successes,fails =  self.particle_beam_search(tokens,device,K)
                else:
                    tokens                  = line.split()
                    results,successes,fails =  self.particle_beam_search(tokens,device,K)
                if results:
                    nll2,df = self.gather_stats(tokens,successes,fails)
                    NLL2   += nll2
                    N      += len(tokens)
                    print(tokens,nll2,len(tokens),np.exp2(nll2/len(tokens)))
                    deriv = RNNGparser.weighted_derivation(results[0])
                    tree  = RNNGparser.deriv2tree(deriv)
                    tree.expand_unaries()
                    if evalb_mode:
                        tree.add_gold_tags(tags)
                    print(str(tree),file=ostream,flush=True)
                    if stats_stream:# writes out the stats
                        #hacked up, but pandas built-in output support for csv  currently hangs on my machine (!?)
                        if stats_header:
                            header = list(df)
                            print('\t'.join(header),file=stats_stream)
                        for row in df.values:
                            print('\t'.join([str(v) for v in row]),file=stats_stream,flush=True)
                        stats_header = False
                else:
                    print('(())',file=ostream,flush=True)
        print(N)
        import pdb;pdb.set_trace()
        print("NLL = %f, PPL = %f"%(NLL2,np.exp2(NLL2/N)),file=sys.stdout)

def train_model(parser,train_treebank,dev_treebank,modelname,device,lr=0.1,epochs=20,batch_size=1,dropout=0.3,parser_params=None):
    """
    Trains a full model for e epochs.
    It minimizes the NLL on the development set with SGD.
    
    Args:
    train_stream (stream): a stream of ConsTree, one on each line
    dev_stream   (stream): a stream of ConsTree, one on each line
    modelname    (string): the dirname of the generated model
    Kwargs:
    lr            (float): the learning rate for SGD
    epochs          (int): the number of epochs to run
    batch_size      (int): the size of the minibatch
    """
    train_time = time.time()
    #Coding & model structure
    #parser.allocate_structure()
    #Training
    trainer      = torch.optim.SGD(parser.parameters(),lr=lr)

    #torch.autograd.set_detect_anomaly(True)
    #trainer      = torch.optim.Adam(parser.parameters(),lr=lr)
    min_nll      = np.inf
    
    ntrain_sentences = len(train_treebank)
    ndev_sentences   = len(dev_treebank)
    
    train_stats = RuntimeStats('NLL','lexNLL','N','lexN')
    valid_stats = RuntimeStats('NLL','lexNLL','N','lexN')

    #print(parser.summary(ntrain_sentences,ndev_sentences,lr,batch_size,epochs))
    for e in range(epochs):

        start_epoch = time.time()
        train_stats.push_row()
        bbegin = 0
        while bbegin < ntrain_sentences:
            bend = min(ntrain_sentences,bbegin+batch_size)
            bef_time = time.time()
            train_stats += parser.eval_sentences(train_treebank[bbegin:bend],device,trainer,backprop=True)
            aft_time = time.time()
            sys.stdout.write('\r===> processed %d training trees; %s'%(bend,format_elapsed(aft_time,train_time)))

            bbegin = bend

        NLL,lex_NLL,N,lexN = train_stats.peek()
        NLL = NLL.cpu()
        lex_NLL = lex_NLL.cpu()
        print('\n[Training]   Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL,np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)

        valid_stats.push_row()
        bbegin = 0
        while bbegin < ndev_sentences:
            bend = min(ndev_sentences,bbegin+batch_size)
            valid_stats += parser.eval_sentences(dev_treebank[bbegin:bend],device,backprop=False)
            bbegin = bend
            
        NLL,lex_NLL,N,lexN = valid_stats.peek()    
        NLL = NLL.cpu()
        lex_NLL = lex_NLL.cpu()
        print('[Validation] Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL, np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)
        print()
        
        if NLL < min_nll:
            parser.save_model(modelname,trainer,e,NLL,parser_params,lr)
        end_epoch = time.time()
        print(f'Total epoch time: {format_elapsed(end_epoch,start_epoch)}')
        print(f'Total elapsed time: {format_elapsed(end_epoch,train_time)}')

    parser.save_model(modelname+'.final',trainer,e,NLL,parser_params,lr)
                                       

def read_config(filename=None):

    """
    Return an hyperparam dictionary
    """
    import configparser
    config = configparser.ConfigParser()
    config.read(filename)

    params = {}
    params['stack_embedding_size'] = int(config['structure']['stack_embedding_size']) if 'stack_embedding_size' in config['structure'] else 100
    params['stack_hidden_size']    = int(config['structure']['stack_hidden_size'])    if 'stack_hidden_size'    in config['structure'] else 100
    params['word_embedding_size']  = int(config['structure']['word_embedding_size'])  if 'word_embedding_size'  in config['structure'] else 100

    params['dropout']         = float(config['learning']['dropout'])      if 'dropout' in config['learning'] else 0.1
    params['learning_rate']   = float(config['learning']['learning_rate'])if 'learning_rate' in config['learning'] else 0.001
    params['num_epochs']      = int(config['learning']['num_epochs'])     if 'num_epochs' in config['learning'] else 20
    params['batch_size']      = int(config['learning']['batch_size'])     if 'batch_size' in config['learning'] else 20
    params['seed']            = int(config['learning']['seed'])           if 'seed' in config['learning'] else 0

    return params
         
if __name__ == '__main__':


    train_file  = ''
    dev_file    = ''
    test_file   = ''
    brown_file  = ''
    model_name  = ''
    config_file = ''
    beam_size   = 400
    stats       = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:d:p:m:b:c:sB:")
    except getopt.GetoptError:
        print('Ooops, wrong command line arguments')
        print('for training...')
        print ('rnnglm.py -t <inputfile> -d <inputfile> -m <model_file> -b <brown_file> -c <config_file>')
        print('for testing...')
        print ('rnnglm.py -m <model_file> -p <test_file> -s')
        sys.exit(0)

    for opt, arg in opts:
        if opt in   ['-t','--train']:
            train_file = arg
        elif opt in ['-d','--dev']:
            dev_file = arg
        elif opt in ['-p','--pred']:
            test_file = arg
        elif opt in ['-c','--config']:
            config_file = arg
        elif opt in ['-m','--model']:
            model_name = arg
        elif opt in ['-b','--brown']:
            brown_file = arg
        elif opt in ['-B','--Beam-size']:
            beam_size  = int(arg)
        elif opt in ['-s','--stats']:
            stats = True

    if train_file and dev_file and brown_file and model_name:
        
        train_stream   = open(train_file)
        dev_stream     = open(dev_file)

        #Trees preprocessing
        train_treebank = []
        for line in train_stream:
            t = ConsTree.read_tree(line)
            ConsTree.strip_tags(t)
            ConsTree.close_unaries(t)
            train_treebank.append(t)

        dev_treebank = []
        for line in dev_stream:
            t = ConsTree.read_tree(line)
            ConsTree.strip_tags(t)
            ConsTree.close_unaries(t)
            dev_treebank.append(t)

        if config_file:
            config = read_config(config_file)

            set_seeds(config['seed'])
            
            parser = RNNGparser(brown_file,\
                                train_treebank,\
                                dev_treebank,\
                                stack_memory_size=config['stack_hidden_size'],\
                                word_embedding_size=config['word_embedding_size'])
            parser.to(device)
            parser_params = {'brown_file': brown_file,
                             'train': train_treebank,
                             'dev': dev_treebank,
                             'stack_size':config['stack_hidden_size'],
                             'word_emb':config['word_embedding_size']}
            train_model(parser,train_treebank,dev_treebank,model_name,device,epochs=config['num_epochs'],lr=config['learning_rate'],batch_size=config['batch_size'],dropout=config['dropout'],parser_params=parser_params)
        else:
            parser = RNNGparser(brown_file,stack_memory_size=200,word_embedding_size=250)
            train_model(parser,train_treebank,dev_treebank,model_name,device,epochs=20,lr=0.1,batch_size=20,parser_params=parser_params)
        train_stream.close()
        dev_stream.close()
        print('\ntraining done.')

    if model_name and test_file:
        
        parser,trainer= RNNGparser.load_model(model_name)
        lstm_initial = (RNNGparser.xavier_init(device,1, parser.stack_hidden_size), RNNGparser.xavier_init(device,1,parser.stack_hidden_size))
        empty_emb = nn.Parameter(torch.randn(1, parser.stack_embedding_size)).to(device)
        parser.rnn.initialize_tensors(lstm_initial,empty_emb)

        test_stream   = open(test_file)
        test_out      = open(model_name+".test.mrg",'w')
        sstream       = open(model_name+'.stats.tsv','w') if stats else None
        evalb_flag    = test_file.endswith('mrg')
        parser.parse_corpus(test_stream,test_out,device,stats_stream=sstream,K=beam_size,evalb_mode=evalb_flag)

        test_out.close()
        test_stream.close()
        if stats:
            sstream.close()
    
