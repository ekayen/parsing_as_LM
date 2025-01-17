#!/usr/bin/env python

import dynet as dy
import configparser,getopt

from collections import namedtuple,Counter

from lexicons  import *
from discotree import *
from proc_monitors import *
from rnng_params import *
   
class BeamElement:
    """
    This class is a place holder for elements in the beam.
    """
    __slots__ = ['prev_element', 'prev_action','prefix_score','configuration']
    
    def __init__(self,prev_element,prev_action,prefix_score):
        """
        Args: 
             prev_element (BeamElement) : the previous element or None
             prev_action       (string) : the action generating this element or None
             prefix_score       (float) : prefix logprobability
        """
        self.prev_element  = prev_element
        self.prev_action   = prev_action
        self.prefix_score  = prefix_score
        self.configuration = None

    @staticmethod
    def init_element(configuration):
        """
        Generates the beam initial (root) element
        Args:
           configuration (tuple): the parser init config
        Returns:
           BeamElement to be used at init
        """
        b = BeamElement(None,None,0.0)
        b.configuration = configuration
        return b
    
    def is_initial_element(self):
        """
        Returns:
            bool. True if the element is root of the beam
        """
        return self.prev_element is None or self.prev_action is None

class StackSymbol:
    """
    A convenience class for symbols on the stack.
    """ 
    def __init__(self,symbol,embedding,predicted=False,sym_range=None):
        """
        Args:
           symbol           (string): a non terminal or a word
           embedding (dy.expression): a dynet expression being the embedding of the subtree dominated by this symbol (or word)
        KwArgs:
           predicted          (bool): predicted or completed
           sym_range           (set): the yield of this symbol as a set of integers
        """
        self.symbol,self.embedding,self.predicted = symbol,embedding,predicted
        self.range = sym_range
        self.has_to_move                              = False

    def is_terminal(self):
        return not self.predicted and len(self.range) == 1
        
    def copy(self): 
        """
        Returns:
           A StackSymbol object, a copy of this symobl
        """
        s = StackSymbol(self.symbol,self.embedding,self.predicted)
        s.has_to_move = self.has_to_move
        s.range       = self.range.copy()
        return s
    
    def schedule_movement(self,flag = True):
        """
        Internal method, flags the symbol for movement
        Args:
            flag (bool) : flags the symbol for movement (True) or stops movement (False) 
        """
        symcopy = self.copy()
        symcopy.has_to_move = flag
        return symcopy

    def complete(self):
        """
        Internal method, setting the symbol as completed
        """
        symcopy = self.copy()
        symcopy.predicted = False
        return symcopy

    def __str__(self):
        s =  '*%s'%(self.symbol,) if self.predicted else '%s*'%(self.symbol,)
        if self.has_to_move:
            s =  'm[%s]'%(s,)        
        return s

def print_config(config):
    S,B,n,stack_state,lab_state,history_state = config 
    return '(%s;%s;%d;%s)'%(','.join([ str(s) for s in S ]),str(B),n,lab_state)
    

class DiscoRNNGparser: 
    """
    This is discontinuous RNNG with pre-order tree traversal and a move action
    """        
    #action codes
    SHIFT           = '<S>'
    OPEN            = '<O>'
    CLOSE           = '<C>'
    TERMINATE       = '<T>'
    MOVE            = '<M>' 

    #labelling states
    WORD_LABEL      = '@w'
    NT_LABEL        = '@n'
    NO_LABEL        = '@'

    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'
    START_POS     = '<START>'
    START_HISTORY = '<START>'
    
    def __init__(self,config_file):
        
        self.read_hyperparams(config_file) 

    def read_hyperparams(self,configfilename):
        #defaults
        self.cond_stack_memory_size     = 128
        self.cond_stack_xsymbol_size    = 128    
        self.cond_word_embedding_size   = 32
        self.pos_embedding_size         = 16
        self.history_xsymbol_size       = 16

        self.gen_stack_memory_size     = 256
        self.stack_xsymbol_size        = 256 
        self.gen_word_embedding_size   = 256
        assert(self.stack_xsymbol_size == self.gen_word_embedding_size)
        
        self.word_dropout          = 0.5

        config = configparser.ConfigParser()
        config.read(configfilename)
        
        self.cond_stack_memory_size     = int(config['conditional']['stack_memory_size'])    #size of stack output
        self.cond_stack_xsymbol_size    = int(config['conditional']['stack_xsymbol_size'])   #size of stack input
        self.cond_word_embedding_size   = int(config['conditional']['word_embedding_size'])
        self.pos_embedding_size         = int(config['conditional']['pos_embedding_size'])
        self.history_xsymbol_size       = int(config['conditional']['history_xsymbol_size'])
 
        self.gen_stack_memory_size     = int(config['generative']['stack_memory_size'])
        self.gen_stack_xsymbol_size    = int(config['generative']['stack_xsymbol_size'])
        self.gen_word_embedding_size   = int(config['generative']['word_embedding_size'])
        self.brown_file                = config['generative']['brown_file']

        
        
    def allocate_conditional_params(self):
        """ 
        This allocates memory for the conditional model parameters
        """
        self.cond_model                     = dy.ParameterCollection()

        #input symbols
        self.cond_nonterminals_embeddings   = self.cond_model.add_lookup_parameters((self.nonterminals.size(),self.cond_stack_xsymbol_size)) 
        self.cond_word_embeddings           = self.cond_model.add_lookup_parameters((self.lexicon.size(),self.cond_word_embedding_size)) 
        self.tag_embeddings                 = self.cond_model.add_lookup_parameters((self.tags.size(),self.pos_embedding_size))
        self.history_embeddings             = self.cond_model.add_lookup_parameters((self.history_syms.size(),self.history_xsymbol_size))
        ## merge words+tags
        self.cond_lex_W                     = self.cond_model.add_parameters((self.cond_stack_xsymbol_size,self.cond_word_embedding_size+self.pos_embedding_size))            
        self.cond_lex_b                     = self.cond_model.add_parameters((self.cond_stack_xsymbol_size))
        
        #stack_lstm
        self.cond_rnn                      = dy.LSTMBuilder(2,self.cond_stack_xsymbol_size, self.cond_stack_memory_size,self.cond_model)   

        
        #output params
        self.cond_structural_W              = self.cond_model.add_parameters((self.actions.size(),self.cond_stack_memory_size*3))         
        self.cond_structural_b              = self.cond_model.add_parameters((self.actions.size())) 
        
        self.cond_nonterminals_W            = self.cond_model.add_parameters((self.nonterminals.size(),self.cond_stack_memory_size*3))   
        self.cond_nonterminals_b            = self.cond_model.add_parameters((self.nonterminals.size()))

        self.cond_move                      = self.cond_model.add_parameters((1,self.cond_stack_memory_size*3))

        #tree embeddings
        self.cond_tree_fwd                  = dy.LSTMBuilder(1,self.cond_stack_xsymbol_size, self.cond_stack_memory_size,self.cond_model)        
        self.cond_tree_bwd                  = dy.LSTMBuilder(1,self.cond_stack_xsymbol_size, self.cond_stack_memory_size,self.cond_model)        
        self.cond_tree_W                    = self.cond_model.add_parameters((self.cond_stack_xsymbol_size,self.cond_stack_memory_size*2))
        self.cond_tree_b                    = self.cond_model.add_parameters((self.cond_stack_xsymbol_size))
 
        #lookahead specific to the cond model
        self.lexer_rnn_bwd                  = dy.LSTMBuilder(1,self.cond_stack_xsymbol_size,self.cond_stack_memory_size,self.cond_model)   
        #history specific to the cond model
        self.history_rnn                    = dy.LSTMBuilder(2,self.history_xsymbol_size,self.cond_stack_memory_size,self.cond_model)   
        
    def allocate_generative_params(self):
        """ 
        This allocates memory for the generative model parameters
        """
        self.gen_model                     = dy.ParameterCollection()

        #input symbols
        self.gen_nonterminals_embeddings   = self.gen_model.add_lookup_parameters((self.nonterminals.size(),self.gen_stack_xsymbol_size)) 
        self.gen_word_embeddings           = self.gen_model.add_lookup_parameters((self.lexicon.size(),self.gen_word_embedding_size)) 

        #stack_lstm
        self.gen_rnn                       = dy.LSTMBuilder(2,self.gen_stack_xsymbol_size, self.gen_stack_memory_size,self.gen_model)     

        #output params
        self.gen_structural_W              = self.gen_model.add_parameters((self.actions.size(),self.gen_stack_memory_size))         
        self.gen_structural_b              = self.gen_model.add_parameters((self.actions.size()))
        
        self.word_softmax                  = dy.ClassFactoredSoftmaxBuilder(self.gen_stack_memory_size,self.brown_file,self.lexicon.words2i,self.gen_model,bias=True)

        self.gen_nonterminals_W            = self.gen_model.add_parameters((self.nonterminals.size(),self.gen_stack_memory_size))   
        self.gen_nonterminals_b            = self.gen_model.add_parameters((self.nonterminals.size()))

        self.gen_move                      = self.gen_model.add_parameters((1,self.gen_stack_memory_size))

        #tree lstm
        self.gen_tree_fwd                  = dy.LSTMBuilder(1,self.gen_stack_xsymbol_size, self.gen_stack_memory_size,self.gen_model)        
        self.gen_tree_bwd                  = dy.LSTMBuilder(1,self.gen_stack_xsymbol_size, self.gen_stack_memory_size,self.gen_model)        
        self.gen_tree_W                    = self.gen_model.add_parameters((self.gen_stack_xsymbol_size,self.gen_stack_memory_size*2))
        self.gen_tree_b                    = self.gen_model.add_parameters((self.gen_stack_xsymbol_size))

    #TRANSITION SYSTEM AND ORACLE
    def init_configuration(self,N,conditional):
        """ 
        Args:
            N            (int): the length of the input 
            conditional (bool): flags for conditional vs generative model

        Inits a starting configuration. A configuration is 5-uple
        S: is the stack
        B: the buffer
        stack_mem: the current state of the stack lstm
        lab_state: the labelling state of the configuration
        
        Args:
           N   (int): the length of the input sequence
        """
        if conditional:
            w0_idx        = self.lexicon.index(DiscoRNNGparser.START_TOKEN)
            w0            = self.cond_word_embeddings[w0_idx] if conditional else self.gen_word_embeddings[w0_idx]        
            t0_idx        = self.tags.index(DiscoRNNGparser.START_POS)
            t0            = self.tag_embeddings[t0_idx]    
            stack_state   = self.cond_rnn.initial_state()
            stack_state   = stack_state.add_input(dy.rectify(self.cond_lex_W*dy.concatenate([w0,t0]) + self.cond_lex_b))
            history_state = self.history_rnn.initial_state()
            hist_embedding = self.history_embeddings[self.history_syms.index(DiscoRNNGparser.START_HISTORY)]
            history_state = history_state.add_input(hist_embedding)
        else:
            w0_idx        = self.lexicon.index(DiscoRNNGparser.START_TOKEN)
            w0            = self.gen_word_embeddings[w0_idx]  
            stack_state   = self.gen_rnn.initial_state()
            stack_state   = stack_state.add_input(w0)
            history_state = None

        return ([ ] ,tuple(range(N)),0, stack_state, DiscoRNNGparser.NO_LABEL,history_state)

    def shift_action(self,configuration):
        """
        This performs a shift action.
        That is the parser commits itself to generate a word at the next step.
        Args:
           configuration (tuple) : a configuration frow where to shift
        Returns: 
           tuple. a configuration resulting from shift 
        """
        S,B,n,stack_state,lab_state,history_state = configuration
        return (S,B,n,stack_state,DiscoRNNGparser.WORD_LABEL,history_state)

    def generate_word(self,configuration,sentence,tag_sequence,conditional):
        """
        This generates a word (performs the actual shifting).
        Args:
           configuration (tuple) :  a configuration frow where to generate a word
           sentence       (list) :  a list of strings, the sentence tokens
           tag_sequence (list)   :  a list of strings, the sentence tags
           conditional     (bool): flag for conditional vs generative model
        Returns:
           tuple. a configuration after word generation
        """
        S,B,n,stack_state,lab_state,history_state = configuration
        shifted_word     = sentence[ B[0] ]
        if conditional:
            shifted_tag   = tag_sequence[ B[0] ]
            wembedding    = self.cond_word_embeddings[self.lexicon.index(shifted_word,alpha_dropout=self.word_dropout)]
            tembedding    = self.tag_embeddings[self.tags.index(shifted_tag)]
            embedding     = dy.concatenate([wembedding,tembedding])
            xinput        = dy.rectify(self.cond_lex_W * embedding + self.cond_lex_b)
            stack_state   = stack_state.add_input(xinput)
            shift_embedding = self.history_embeddings[self.history_syms.index(DiscoRNNGparser.SHIFT)]
            history_state = history_state.add_input(shift_embedding)
            return (S + [StackSymbol(B[0],xinput,predicted=False,sym_range=[B[0]])],B[1:],n,stack_state,DiscoRNNGparser.NO_LABEL,history_state)
        else: 
            embedding   = self.gen_word_embeddings[self.lexicon.index(shifted_word)]
            stack_state = stack_state.add_input(embedding)
            return (S + [StackSymbol(B[0],embedding,predicted=False,sym_range=[B[0]])],B[1:],n,stack_state,DiscoRNNGparser.NO_LABEL,None)
 
    def open_action(self,configuration):
        """
        Args:
           configuration (tuple): a configuration
        Returns:
           A configuration
        """
        S,B,n,stack_state,lab_state,history_state = configuration
        return (S,B,n,stack_state,DiscoRNNGparser.NT_LABEL,history_state) 
    
    def open_nonterminal(self,configuration,label,conditional):
        """
        The nonterminal labelling action. This adds an open nonterminal on the stack under the stack top (left corner style inference)
        
        Arguments:
            configuration (tuple) : a configuration where to perform the labelling
            label         (string): the nonterminal label
            conditional     (bool): flag for conditional vs generative model
        Returns:
            tuple. A configuration resulting from the labelling
        """
        S,B,n,stack_state,lab_state,history_state = configuration
        nt_idx        = self.nonterminals.index(label)
        embedding     = self.cond_nonterminals_embeddings[nt_idx] if conditional else self.gen_nonterminals_embeddings[nt_idx]
        stack_state   = stack_state.add_input(embedding)
        if conditional:
            openX_embedding = self.history_embeddings[self.history_syms.index(label)]
            history_state = history_state.add_input(openX_embedding)
        return (S + [StackSymbol(label,embedding,predicted=True,sym_range=[B[0]])],B,n + 1,stack_state,DiscoRNNGparser.NO_LABEL,history_state) 

    def close_action(self,configuration,conditional): 
        """
        This actually executes the RNNG CLOSE action.
        Args:
           configuration (tuple): a configuration frow where to perform open
           conditional    (bool): flag for conditional vs generative model
        Returns:
           tuple. A configuration resulting from closing the constituent
        """
        S,B,n,stack_state,lab_state,history_state = configuration
        newS = S[:]
        closed_symbols = [ ]
        moved_symbols  = [ ]
        complete_range = set() 

        while not (newS[-1].predicted and not newS[-1].has_to_move):
            
            stack_state = stack_state.prev()
            symbol = newS.pop() 
            if symbol.has_to_move:
                symbol = symbol.schedule_movement(False)
                moved_symbols.append(symbol)
            else:
                closed_symbols.append(symbol)
                if symbol.range:
                    complete_range = complete_range | set(symbol.range)

        stack_state = stack_state.prev()      
        completeNT = newS.pop()
        
        #computes the tree embedding of the completed stuff
        nt_idx    = self.nonterminals.index(completeNT.symbol)

        fwd_state = self.cond_tree_fwd.initial_state() if conditional else self.gen_tree_fwd.initial_state()
        fwd_state = fwd_state.add_input(self.cond_nonterminals_embeddings[nt_idx]) if conditional else  fwd_state.add_input(self.gen_nonterminals_embeddings[nt_idx])
        for SYM in reversed(closed_symbols):
            fwd_state = fwd_state.add_input(SYM.embedding)
            
        bwd_state = self.cond_tree_bwd.initial_state() if conditional else self.gen_tree_bwd.initial_state()
        bwd_state = bwd_state.add_input(self.cond_nonterminals_embeddings[nt_idx]) if conditional else bwd_state.add_input(self.gen_nonterminals_embeddings[nt_idx])
        for SYM in closed_symbols:
            bwd_state = bwd_state.add_input(SYM.embedding)

        tree_h         = dy.concatenate([fwd_state.output(),bwd_state.output()])
        W = self.cond_tree_W if conditional else self.gen_tree_W
        b = self.cond_tree_b if conditional else self.gen_tree_b
        tree_embedding = dy.tanh(W * tree_h + b)
        
        completeNT           = completeNT.complete()
        completeNT.range     = complete_range 
        completeNT.embedding = tree_embedding
        newS.append(completeNT)
        stack_state = stack_state.add_input(tree_embedding)
         
        #updates the stack state when putting back the moved elements
        newS.extend(reversed(moved_symbols))
        new_n = n-1
        for SYM in reversed(moved_symbols):
            stack_state = stack_state.add_input(SYM.embedding)
            if SYM.predicted:
                new_n += 1
        if conditional:
            close_embedding = self.history_embeddings[self.history_syms.index(DiscoRNNGparser.CLOSE)]
            history_state = history_state.add_input(close_embedding)
        return (newS,B,new_n,stack_state,DiscoRNNGparser.NO_LABEL,history_state) 

        
    def move_action(self,configuration,stack_idx,conditional):
        """
        This actually schedules a symbol down in the stack for movement
        Args:
           configuration (tuple) : a configuration
           stack_idx       (int) : the index in the stack of the element to move (top has index 0) 
        Returns:
           Tuple. A configuration resulting from moving the constituent
        """
        S,B,n,stack_state,lab_state,history_state = configuration
        newS = S[:]
        moved_elt =  newS[-stack_idx-1]
        newS[-stack_idx-1] = moved_elt.schedule_movement(True)
        new_n = n-1 if moved_elt.predicted else n
        if conditional:
            move_embedding = self.history_embeddings[self.history_syms.index(DiscoRNNGparser.MOVE)]
            history_state = history_state.add_input(move_embedding)
        return (newS,B,new_n,stack_state,DiscoRNNGparser.NO_LABEL,history_state)

    def static_oracle(self,ref_root,global_root,sentence,tag_sequence,conditional,configuration=None):
        """
        Generates a list of configurations and returns a list of actions to exec given a ref tree
        Args: 
          ref_root    (DiscoTree): the local root reference node.
          global_root (DiscoTree): the global root of the reference tree.
          sentence         (list): a list  of strings, the tokens
        KwArgs:
          configuration   (tuple): a configuration tuple or None at init
        Returns: 
          A list of actions, the last configuration.
        """       
        def occurs_predicted(ref_node,configuration):  # (occur check #1)
            #returns True if predicted node already on the stack
            S,B,n,stack_state,lab_state,history_state = configuration
            lc_idx = ref_node.left_corner() 
            for node in reversed(S): 
                if not node.predicted and min(node.range) == lc_idx: 
                    return True 
            return False
         
        def occurs_completed(ref_node,configuration):  # (occur check #2)
            #returns True if completed node already on the stack

            S,B,n,stack_state,lab_state,history_state = configuration
            for elt in S:
                if not elt.predicted and ref_node.is_dominated_by(elt.range):
                    return True
            return False
        
        if configuration is None:                       #init
            N = len(ref_root.words()) 
            configuration = self.init_configuration(N,conditional)   

        if ref_root.is_leaf(): 
            configuration = self.shift_action(configuration)
            configuration = self.generate_word(configuration,sentence,tag_sequence,conditional)
            S,B,n,stack_state,lab_state,history_state = configuration
            sh_word                                   = sentence[S[-1].symbol]
            act_list                                  = [DiscoRNNGparser.SHIFT,sh_word]
            return (act_list, configuration) 
        else:
            ##Recursive processing
            #A. Root 
            act_list = []
            if not occurs_predicted(ref_root,configuration):
                configuration = self.open_action(configuration) 
                configuration = self.open_nonterminal(configuration,ref_root.label,conditional)
                act_list.extend([DiscoRNNGparser.OPEN,ref_root.label])
                
            #B. Recursive calls 
            for child in ref_root.covered_nodes(global_root):
                if not occurs_completed(child,configuration): #occur check : recursive call if and only if not already in
                    #non local extra processing
                    local = ref_root.dominates(child.range) 
                    if not local: 
                        for ancestor in global_root.get_lc_ancestors(child.range):
                            if ancestor is not child:
                                configuration = self.open_action(configuration) 
                                configuration = self.open_nonterminal(configuration,ancestor.label,conditional)
                                act_list.extend([DiscoRNNGparser.OPEN,ancestor.label]) 

                    local_actions, configuration = self.static_oracle(child,global_root,sentence,tag_sequence,conditional,configuration)
                    act_list.extend(local_actions)


            #C. Perform moves
            S,B,n,stack_state,lab_state,history_state = configuration
            for stack_idx, stack_elt in enumerate(reversed(S)):
                local = ref_root.dominates(stack_elt.range)
                if stack_elt.predicted and local:
                    break
                if not local:
                    #assert(stack_idx > 0) -> nope. there exist cases where this assertion does not hold (!)
                    configuration = self.move_action(configuration,stack_idx,conditional)
                    act_list.extend([DiscoRNNGparser.MOVE,stack_idx])       

            #D. Close
            configuration = self.close_action(configuration,conditional)
            act_list.append(DiscoRNNGparser.CLOSE)
            if ref_root is global_root:
                act_list.append(DiscoRNNGparser.TERMINATE)
            return (act_list,configuration)

        
    def deriv2tree(self,derivation): 
        """
        Generates a discontinuous tree from the derivation
        Args:
           derivation (list): a list of actions as strings
        Returns:
          DiscoTree. The root of the tree
        """
        StackElt = namedtuple('StackElt',['symbol','predicted','has_to_move'])

        stack = []  
        inc_index   = 0
        prev_action = None
        for action in derivation:
            if prev_action   == DiscoRNNGparser.SHIFT:
                stack.append( StackElt(symbol=DiscoTree(action,child_index = inc_index), predicted=False,has_to_move=False) )
                inc_index += 1
                
            elif prev_action == DiscoRNNGparser.OPEN: 
                stack.append( StackElt(symbol=action, predicted=True,has_to_move=False) ) 
                
            elif prev_action == DiscoRNNGparser.MOVE: #sets the move on the stack
                sym,pred,mov = stack[ -int(action)-1 ]
                stack[ -int(action)-1 ] = StackElt(symbol=sym,predicted=pred,has_to_move=True)
                
            elif action == DiscoRNNGparser.CLOSE:
                nmoves   = 0
                children = [ ]
                moved    = [ ]
                while stack:
                    node = stack.pop()
                    if node.predicted and not node.has_to_move:
                        stack.append( StackElt(symbol=DiscoTree(node.symbol,children),predicted=False,has_to_move=False) )
                        stack.extend(reversed(moved) ) 
                        break 
                    if node.has_to_move:
                        sym,pred,mov = node
                        moved.append( StackElt(symbol=sym,predicted=pred,has_to_move=False)  )
                    else:
                        children.append(node.symbol)
            prev_action = action
            
        root = stack.pop()
        return root.symbol

    def code_lexicon(self,train_words,train_tags):
        """
        Builds indexes for word symbols found in the treebank
        """        
        known_words = Counter()
        known_tags  = set([ ])
        for words,tags in zip(train_words,train_tags):
            known_words.update(words)
            known_tags.update(tags)
            
        known_words.update([DiscoRNNGparser.START_TOKEN])
        self.brown_file  = normalize_brown_file(self.brown_file,set(known_words.keys()),self.brown_file+'.unk',UNK_SYMBOL=DiscoRNNGparser.UNKNOWN_TOKEN)
        self.lexicon     = SymbolLexicon(known_words,unk_word=DiscoRNNGparser.UNKNOWN_TOKEN)

        known_tags.add(DiscoRNNGparser.START_POS)
        self.tags = SymbolLexicon( list(known_tags))
                    
        return self.lexicon,self.tags
     
    def code_nonterminals(self,train_treebank,dev_treebank):
        """
        Extracts the nonterminals from a treebank and codes them on integers as a lexicon object.
        
        Args:
           train_treebank   (list) : a list of trees  where to extract the non terminals from
           dev_treebank     (list) : a list of trees  where to extract the non terminals from
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
        #Structural actions are coded on the first slots
        self.actions         = SymbolLexicon([DiscoRNNGparser.SHIFT,DiscoRNNGparser.OPEN,DiscoRNNGparser.CLOSE,DiscoRNNGparser.TERMINATE])
        self.history_syms    = SymbolLexicon(self.nonterminals.i2words+[DiscoRNNGparser.SHIFT,\
                                                                        DiscoRNNGparser.CLOSE,\
                                                                        DiscoRNNGparser.MOVE,\
                                                                        DiscoRNNGparser.START_HISTORY,\
                                                                        DiscoRNNGparser.TERMINATE])
        
        return self.actions
 
    def allowed_structural_actions(self,configuration): 
        """
        Returns the list of structural actions allowed given this configuration.
        Arguments:
           configuration          (tuple): a configuration
        Returns:
           a list. Indexes of the allowed actions
        """
        S,B,n,stack_state,lab_state,history_state = configuration
        #Analyze stack top
        children             = []
        children_terminals   = 0
        for idx,stack_elt in enumerate(reversed(S)):
            children.append((not stack_elt.predicted) and (not stack_elt.has_to_move))
            children_terminals += (stack_elt.is_terminal() and (not stack_elt.has_to_move))
            if stack_elt.predicted and not stack_elt.has_to_move:
                break
        if children:         #! last element is always predicted ! 
            children[-1] = False 
 
        allowed_static= [False] * self.actions.size() 
        
        if B and n <= 100:
            allowed_static[self.actions.index(DiscoRNNGparser.OPEN)]      = True  
        if (B and n >= 1):
            allowed_static[self.actions.index(DiscoRNNGparser.SHIFT)]     = True  
        if not B and n == 0 and len(S) == 1:
            allowed_static[self.actions.index(DiscoRNNGparser.TERMINATE)] = True

        sum_children = sum(children)
        if (n >= 2 and sum_children >= 2) or (n >=2 and sum_children == 1 and children_terminals == 1) or (n==1 and not B) :
            allowed_static[self.actions.index(DiscoRNNGparser.CLOSE)]     = True  

        if n >= 1:
            allowed_dynamic = children
            if n >= 2:
                allowed_dynamic[-1] = True
        else:
            allowed_dynamic = [False]*len(children)
        
            
        allowed_idxes = [idx for idx, mask_val in enumerate(allowed_static+allowed_dynamic) if mask_val]
        return allowed_idxes
    
    def ifdropout(self,expression):
        """
        Applies dropout to a dynet expression only if dropout > 0.0.
        """
        return dy.dropout(expression,self.dropout) if self.dropout > 0.0 else expression

    def dynamic_move_matrix(self,stack,stack_state,history_state,buffer_embedding,conditional):
        """
        Dynamically computes the score of each possible valid move action
        Args:
            stack                       (list): a list of StackELements (the stack of a configuration)
            stack_state          (dynet stuff): pointer to an rnn state
            buffer_embedding(dynet expression): the embedding of the first word in the buffer
            conditional                (bool) : bool stating if we use conditional or generative params
        Returns: 
            A dynet expression
        """
        local_state  = stack_state
        stack_scores = [ ] 
        
        for idx,stack_elt in enumerate(reversed(stack)):
            if conditional:
                H =  dy.concatenate([local_state.output(),history_state.output(),buffer_embedding])
                stack_scores.append( self.cond_move * dy.tanh(H) )
            else:
                stack_scores.append( self.gen_move * dy.tanh(local_state.output()))
            if stack_elt.predicted and not stack_elt.has_to_move : #check this condition:up to where can we move ?
                break
            local_state = local_state.prev() 

        return dy.concatenate(stack_scores) if stack_scores else stack_scores
    
    def predict_action_distrib(self,configuration,sentence,word_encodings,conditional):
        """
        Predicts the log distribution for next actions from the current configuration.
        Args:
          configuration           (tuple): the current configuration
          sentence                 (list): a list of string, the tokens
          word_encodings           (list): a list of embeddings for the tokens. None in case of generative inference
          conditional              (bool): flag stating whether to perform conditional or generative inference
        Returns:
            a list of couples (action, log probability). The list is empty if the parser is trapped (aka no action is possible).
            currently returns a zip generator.
        """
        def code2action(act_idx): 
            return (self.MOVE,act_idx-self.actions.size())  if act_idx >= self.actions.size() else  self.actions.wordform(act_idx)
     
        S,B,n,stack_state,lab_state,history_state = configuration

        if lab_state == DiscoRNNGparser.WORD_LABEL:
            next_word     = (sentence[B[0]])
            if conditional:    
                return [(next_word,0)] # in the discriminative case words are given and have prob = 1.0
            else:
                next_word_idx = self.lexicon.index(next_word) 
                return [(next_word,-self.word_softmax.neg_log_softmax(stack_state.output(),next_word_idx).value())]
        elif lab_state == DiscoRNNGparser.NT_LABEL:
            if conditional:
                word_idx = B[0] if B else -1
                H = dy.concatenate([stack_state.output(),history_state.output(),word_encodings[word_idx]])
                logprobs = dy.log_softmax(self.cond_nonterminals_W  * dy.tanh(H)  + self.cond_nonterminals_b).value()
                return zip(self.nonterminals.i2words,logprobs)
            else:
                H = stack_state.output()
                logprobs = dy.log_softmax(self.gen_nonterminals_W  * dy.tanh(H)  + self.gen_nonterminals_b).value()
                return zip(self.nonterminals.i2words,logprobs)
            
        elif lab_state == DiscoRNNGparser.NO_LABEL :
            restr_mask       = self.allowed_structural_actions(configuration)
            if conditional:                
                if restr_mask:
                    word_idx         = B[0] if B else -1
                    buffer_embedding = word_encodings[word_idx]
                    hidden_input     = dy.concatenate([stack_state.output(),history_state.output(),buffer_embedding])
                    static_scores    = self.cond_structural_W  * dy.tanh(hidden_input)  + self.cond_structural_b
                    move_scores      = self.dynamic_move_matrix(S,stack_state,history_state,buffer_embedding,True)
                    all_scores       = dy.concatenate([static_scores,move_scores]) if move_scores else static_scores
                    logprobs         = dy.log_softmax(all_scores,restr_mask).value()                     
                    return [ (code2action(action_idx),logprob) for action_idx,logprob in enumerate(logprobs) if action_idx in restr_mask]
                #parser trapped, let it crash
            else:
                if restr_mask: 
                    hidden_input     = stack_state.output()
                    static_scores    = self.gen_structural_W  * dy.tanh(hidden_input)  + self.gen_structural_b
                    move_scores      = self.dynamic_move_matrix(S,stack_state,history_state,None,False)
                    all_scores       = dy.concatenate([static_scores,move_scores]) if move_scores else static_scores
                    logprobs         = dy.log_softmax(all_scores,restr_mask).value()                     
                    return [ (code2action(action_idx),logprob) for action_idx,logprob in enumerate(logprobs) if action_idx in restr_mask]
                #parser trapped, let it crash
        return [ ]
 
     
    def eval_action_distrib(self,configuration,sentence,word_encodings,ref_action,conditional): 
        """
        Evaluates the model predictions against the reference data.
        Args:
          configuration   (tuple): the current configuration
          sentence         (list): a list of string, the tokens
          word_encodings   (list): a list of embeddings for the tokens. None in case of generative inference
          ref_action     (string): the reference action.
          conditional              (bool): flag stating whether to perform conditional or generative inference
        Returns: 
            a dynet expression. The loss (NLL) for this action
        """        
        def code2action(act_idx): 
            return (self.MOVE,act_idx-self.actions.size())  if act_idx >= self.actions.size() else  self.actions.wordform(act_idx)
        
        S,B,n,stack_state,lab_state,history_state = configuration

        if lab_state == DiscoRNNGparser.WORD_LABEL:
            if conditional:
                nll = dy.scalarInput(0.0)  #in the discriminative case the word is given and has nll = 0
            else:
                ref_idx  = self.lexicon.index(ref_action)
                nll = self.word_softmax.neg_log_softmax(stack_state.output(),ref_idx)

        elif lab_state == DiscoRNNGparser.NT_LABEL:
            
            ref_idx  = self.nonterminals.index(ref_action)
            if conditional:
                word_idx = B[0] if B else -1
                H        = dy.concatenate([stack_state.output(),history_state.output(),word_encodings[word_idx]])
                nll      = dy.pickneglogsoftmax(self.cond_nonterminals_W  * self.ifdropout(dy.tanh(H)) + self.cond_nonterminals_b,ref_idx)
            else: 
                H   = stack_state.output()
                nll = dy.pickneglogsoftmax(self.gen_nonterminals_W  * dy.tanh(H)  + self.gen_nonterminals_b,ref_idx)
            
        elif lab_state == DiscoRNNGparser.NO_LABEL:

            restr_mask       = self.allowed_structural_actions(configuration)
            ref_idx          = self.actions.size() + ref_action if type(ref_action) == int else self.actions.index(ref_action)
            
            if conditional:
                word_idx         = B[0] if B else -1
                buffer_embedding = word_encodings[word_idx]
                hidden_input     = dy.concatenate([stack_state.output(),history_state.output(),buffer_embedding])
                static_scores    = self.cond_structural_W  * self.ifdropout(dy.tanh(hidden_input))  + self.cond_structural_b
                move_scores      = self.dynamic_move_matrix(S,stack_state,history_state,buffer_embedding,conditional)
                all_scores        = dy.concatenate([static_scores,move_scores]) if move_scores else static_scores
                nll              = -dy.pick(dy.log_softmax(all_scores,restr_mask),ref_idx)
            else:
                 hidden_input     = stack_state.output()
                 static_scores    = self.gen_structural_W  * dy.tanh(hidden_input)  + self.gen_structural_b
                 move_scores      = self.dynamic_move_matrix(S,stack_state,history_state,None,conditional)
                 all_scores       = dy.concatenate([static_scores,move_scores]) if move_scores else static_scores
                 nll              = -dy.pick(dy.log_softmax(all_scores,restr_mask),ref_idx)                    
        else: 
            print('error in evaluation')
        return nll
 
    def eval_derivation(self,ref_derivation,sentence,tag_sequence,word_encodings,conditional,backprop=True):
        """
        Evaluates the model predictions against the reference derivation

        Args:
          ref_derivation                (list) : a reference derivation
          sentence                      (list) : a list of strings (words)
          tag_sequence                  (list) : a list of strings (tags)
          word_encodings                (list) : a list of dynet expressions (word embeddings)
          conditional                   (bool) : a flag telling if we train a conditional or a generative model
        Kwargs:
          backprop                       (bool): a flag telling if we perform backprop or not
        Returns:
          RuntimeStats. the model NLL, the word only NLL, the size of the derivations, the number of predicted words 
        """
        runstats = RuntimeStats('NLL','lexNLL','N','lexN')
        runstats.push_row() 
        
        runstats['lexN'] = len(sentence)
        runstats['N']    = len(ref_derivation)
 
        all_NLL     = [] #collects the local losses in the batch
        lexical_NLL = [] #collects the local losses in the batch (for word prediction only)
        
        configuration = self.init_configuration( len(sentence),conditional ) 
        prev_action = None
        for ref_action in ref_derivation:
            S,B,n,stack_state,lab_state,history_state = configuration                

            if ref_action ==  DiscoRNNGparser.MOVE: #skips the move
                prev_action = ref_action
                continue
            
            nll =  self.eval_action_distrib(configuration,sentence,word_encodings,ref_action,conditional)
            all_NLL.append( nll )
            
            if lab_state == DiscoRNNGparser.WORD_LABEL:
                configuration = self.generate_word(configuration,sentence,tag_sequence,conditional)
                lexical_NLL.append( nll )
            elif lab_state == DiscoRNNGparser.NT_LABEL:
                configuration = self.open_nonterminal(configuration,ref_action,conditional)
            elif prev_action == DiscoRNNGparser.MOVE:
                configuration = self.move_action(configuration,int(ref_action),conditional)
            elif ref_action == DiscoRNNGparser.CLOSE:
                configuration = self.close_action(configuration,conditional)
            elif ref_action == DiscoRNNGparser.OPEN:
                configuration = self.open_action(configuration)
            elif ref_action == DiscoRNNGparser.SHIFT:
                configuration = self.shift_action(configuration)
            elif ref_action == DiscoRNNGparser.TERMINATE:
                pass
            prev_action = ref_action

        loss     = dy.esum(all_NLL) 
        lex_loss = dy.esum(lexical_NLL)
        
        runstats['NLL']    = loss.value() 
        runstats['lexNLL'] = lex_loss.value() 
         
        if backprop:
            loss.backward()
            try:
                self.trainer.update()
            except RuntimeError:
                print('\nGradient exploded, batch update aborted...')        
        return runstats

    def rescore_derivation(self,base_derivation,sentence):
        """
        The generative model is used to rescore this derivation.
        """
        r_derivation = [ ]
        prefix_score = 0.0
        
        prev_action = None
        configuration = self.init_configuration( len(sentence),conditional )
        
        for action,prob in base_derivation:
            S,B,n,stack_state,lab_state,history_state = configuration
            if action ==  DiscoRNNGparser.MOVE: #skips the move
                prev_action = action
                continue
            
            gen_prob = self.eval_action_distrib(configuration,sentence,None,action,False)
            prefix_score += gen_prob
    
            r_derivation.append( (action,prefix_score) )
            
            if lab_state == DiscoRNNGparser.WORD_LABEL:
                configuration = self.generate_word(configuration,sentence,None,False)
            elif lab_state == DiscoRNNGparser.NT_LABEL:
                configuration = self.open_nonterminal(configuration,action,False)
            elif prev_action == DiscoRNNGparser.MOVE:
                configuration = self.move_action(configuration,int(action),conditional)
            elif ref_action == DiscoRNNGparser.CLOSE:
                configuration = self.close_action(configuration,False)
            elif ref_action == DiscoRNNGparser.OPEN:
                configuration = self.open_action(configuration)
            elif ref_action == DiscoRNNGparser.SHIFT:
                configuration = self.shift_action(configuration)
            elif ref_action == DiscoRNNGparser.TERMINATE:
                pass 
            prev_action = ref_action
             
        return r_derivation
  
    def encode_words(self,sentence,pos_sequence):
        """
        Runs a backward LSTM on the input sentence.
        Used by the conditional model to get a lookahead
        Args:
           sentence (list): a list of strings
        Returns list. A list of dynet expressions (the encoded words)
        """
        #init
        wembedding      = self.cond_word_embeddings[ self.lexicon.index(DiscoRNNGparser.START_TOKEN) ]
        tembedding      = self.tag_embeddings[ self.tags.index(DiscoRNNGparser.START_POS) ]        
        lex_state       = self.lexer_rnn_bwd.initial_state()
        lex_state       = lex_state.add_input( self.cond_lex_W*dy.concatenate([wembedding,tembedding])+ self.cond_lex_b )
        
        #recurrence 
        word_embeddings = [self.cond_word_embeddings[self.lexicon.index(word,alpha_dropout=self.word_dropout)] for word in reversed(sentence) ]
        tag_embeddings  = [self.tag_embeddings[self.tags.index(pos)] for pos in reversed(pos_sequence) ]

        xinput          = [self.cond_lex_W*dy.concatenate([wembedding,tembedding]) + self.cond_lex_b for wembedding,tembedding in zip(word_embeddings,tag_embeddings)]
        word_encodings  = lex_state.transduce(xinput)
        word_encodings.reverse()
        return word_encodings

    def eval_sentence(self,ref_tree,ref_tags,conditional,backprop=True):
        #add an option for training the generative here
        """
        Evaluates the model predictions against the reference data.
        and optionally performs backpropagation. 
        The function either takes a single tree or a batch of trees (as list) for evaluation.
        Args:
          ref_tree                   (ConsTree): a reference tree or a single tree.
        Kwargs:
          conditional                    (bool): a flag telling if we use the conditional or the generative model
          backprop                       (bool): a flag telling if we perform backprop
        Returns:
          RuntimeStats. the model NLL, the word only NLL, the size of the derivations, the number of predicted words on this batch
        """
        dy.renew_cg()

        dropout,word_dropout = self.dropout,self.word_dropout
        
        if not backprop:
            self.dropout,self.word_dropout = 0.0,0.0
              
        sentence = ref_tree.words()
        if conditional:
            word_encodings         = self.encode_words(sentence,ref_tags)
            derivation,last_config = self.static_oracle(ref_tree,ref_tree,sentence,ref_tags,conditional)
            stats = self.eval_derivation(derivation,sentence,ref_tags,word_encodings,conditional,backprop)
        else:
            word_encodings         = None
            derivation,last_config = self.static_oracle(ref_tree,ref_tree,sentence,None,conditional)
            stats = self.eval_derivation(derivation,sentence,None,word_encodings,conditional,backprop)

        if not backprop:
            self.dropout,self.word_dropout = dropout,word_dropout
        return stats
            
    @staticmethod
    def prune_beam(beam,K): 
        """
        Prunes the beam to the top K elements using the *discriminative* probability (performs a K-Argmax).
        Inplace destructive operation on the beam.
        Args:
             beam  (list) : a beam data structure
             K       (int): the number of elts to keep in the Beam
        Returns:
             The beam object
        """
        beam.sort(key=lambda x:x.prefix_score,reverse=True)
        beam = beam[:K]
        return beam
 
    def exec_beam_action(self,beam_elt,sentence,tag_sequence,conditional):
        """
        Generates the element's configuration and assigns it internally.
        Args:
             beam_elt  (BeamElement): a BeamElement missing its configuration
             sentence         (list): a list of strings, the tokens.
             tag_sequence     (list): a list of strings, the tags.
        """
        
        if  beam_elt.is_initial_element():
            beam_elt.configuration = self.init_configuration(len(sentence),conditional)
        else:
            configuration = beam_elt.prev_element.configuration
            S,B,n,stack_state,lab_state,history_state = configuration
            
            if lab_state == DiscoRNNGparser.WORD_LABEL:
                beam_elt.configuration = self.generate_word(configuration,sentence,tag_sequence,conditional)
            elif lab_state == DiscoRNNGparser.NT_LABEL:
                beam_elt.configuration = self.open_nonterminal(configuration,beam_elt.prev_action,conditional)
            elif type(beam_elt.prev_action) == tuple :
                move_label,mov_idx = beam_elt.prev_action
                beam_elt.configuration = self.move_action(configuration,mov_idx,conditional) 
            elif beam_elt.prev_action == DiscoRNNGparser.CLOSE:
                beam_elt.configuration = self.close_action(configuration,conditional)
            elif beam_elt.prev_action == DiscoRNNGparser.OPEN:
                beam_elt.configuration = self.open_action(configuration)
            elif beam_elt.prev_action == DiscoRNNGparser.SHIFT:
                beam_elt.configuration = self.shift_action(configuration)
            elif beam_elt.prev_action == DiscoRNNGparser.TERMINATE:
                beam_elt.configuration = configuration
            else:
                print('oops')
        
    def predict_beam(self,sentence,K,tag_sequence):
        """
        Performs parsing and returns an ordered list of successful beam elements.
        Discriminative version.

        The default search strategy amounts to sample the search space with discriminative probs and to rank the succesful states with generative probs.
        The alternative search strategy amounts to explore the search space with a conventional K-argmax pruning method (on disc probs) and to rank the results with generative probs.
        Args: 
              sentence      (list): list of strings (tokens)
              K              (int): beam width
              tag_sequence  (list): a list of strings (tags)
        Returns:
             list. List of BeamElements.
        """        
        dy.renew_cg()
        
        word_encodings = self.encode_words(sentence,tag_sequence) if tag_sequence else None
         
        init = BeamElement.init_element(self.init_configuration(len(sentence),True))
        beam,successes  = [init],[ ] 

        while beam :
            beam = DiscoRNNGparser.prune_beam(beam,K) #pruning
            for elt in beam:
                self.exec_beam_action(elt,sentence,tag_sequence,True) #lazily builds configs
            next_preds = [ ] 
            for elt in beam:
                configuration = elt.configuration
                S,B,n,stack_state,lab_state,history_state = configuration
                if lab_state == DiscoRNNGparser.WORD_LABEL:
                    for (action, logprob) in self.predict_action_distrib(configuration,sentence,word_encodings,True):
                        #print(action,exp(logprob))
                        next_preds.append(BeamElement(elt,action,elt.prefix_score+logprob))
                elif lab_state == DiscoRNNGparser.NT_LABEL:
                    for (action, logprob) in self.predict_action_distrib(configuration,sentence,word_encodings,True):
                        #print(action,exp(logprob))
                        next_preds.append(BeamElement(elt,action,elt.prefix_score+logprob))
                else:
                    for (action, logprob) in self.predict_action_distrib(configuration,sentence,word_encodings,True):
                        #print(action,exp(logprob))
                        if action == DiscoRNNGparser.TERMINATE:
                            successes.append(BeamElement(elt,action,elt.prefix_score+logprob))
                        else: 
                            next_preds.append(BeamElement(elt,action,elt.prefix_score+logprob))
            beam = next_preds 
        if successes:
            successes.sort(key=lambda x:x.prefix_score,reverse=True)
            successes = successes[:K]
        return successes

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
        
        dy.renew_cg()
        
        init = BeamElement.init_element(self.init_configuration(len(sentence),False))
        next_word,successes  = [init],[ ]
        
        while next_word:
            
            this_word = next_word
            next_word = [ ]            
            while this_word and len(next_word) < K:
                    fringe     = [ ]
                    fast_track = [ ]
                    for elt in this_word:
                        configuration = elt.configuration
                        for (action, logprob) in self.predict_action_distrib(configuration,sentence,None,None):
                            if elt.prev_action == DiscoRNNGparser.SHIFT: 
                                new_elt = BeamElement(elt,action,elt.prefix_score+logprob)
                                fast_track.append(new_elt)
                            else:
                                new_elt = BeamElement(elt,action,elt.prefix_score+logprob)
                                fringe.append(new_elt)
                                
                    fast_track.sort(key=lambda x:x.prefix_score,reverse=True)
                    fast_track = fast_track[:Kft]
                    fringe.sort(key=lambda x:x.prefix_score,reverse=True)
                    fringe = fringe[:K-len(fast_track)]+fast_track
                    
                    this_word = [ ]
                    for s in fringe:
                        prev_prev_action    = s.prev_element.prev_action
                        if prev_prev_action == DiscoRNNGparser.SHIFT: #<=> tests if we currently generate a word
                            next_word.append(s)
                        elif s.prev_action ==  DiscoRNNGparser.TERMINATE:
                            successes.append(s)
                        else:
                            self.exec_beam_action(s,sentence,None,False)
                            this_word.append(s)
                            
            next_word.sort(key=lambda x:x.prefix_score,reverse=True)
            next_word = next_word[:Kw]
            for elt in next_word:
                self.exec_beam_action(elt,sentence,None,False)
        if successes:
            successes.sort(key=lambda x:x.prefix_score,reverse=True)
            successes = successes[:K]
        return successes

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
            if type(current.prev_action) == tuple:
                mov_code,mov_idx = current.prev_action
                D.append((mov_idx,current.prefix_score))
                D.append((mov_code,current.prefix_score))
            else:
                D.append((current.prev_action,current.prefix_score))
            current = current.prev_element
        D.reverse()   
        return D
        
    def parse_corpus(self,istream,ostream=sys.stdout,evalb_mode=False,stats_stream=None,K=5,kbest=1,conditional=True,generative=True):
        """ 
        Parses a corpus and prints out the trees in a file.
        Args: 
           istream  (stream): the stream where to read the data from
           ostream  (stream): the stream where to write the data to
           evalb_mode (bool): a bool stating if we reinsert tags provided by the ref tree into the output to ensure compat with evalb
        Kwargs: 
           stats_stream (string): the stream where to dump stats
           K               (int): the size of the beam
           kbest           (int): the number of parses hypotheses outputted per sentence (<= K)
        """         
        self.dropout,self.word_dropout  = 0.0,0.0
        NLL = 0
        N   = 0
        stats_header = True 
        for line in istream:             
            tree        = DiscoTree.read_tree(line)
            tag_nodes   = tree.pos_nodes()
            tokens      = [x.children[0].label for x in tag_nodes]
            tags        = [x.label for x in tag_nodes]
            results     = self.predict_beam(tokens,K,tags) if conditional else self.predict_beam_generative(tokens,K)
            derivations = [DiscoRNNGparser.weighted_derivation(r) for r in results]
            
            if conditional and generative and derivations:                 #reranks the derivations with the generative model
                derivations = [self.rescore_derivation(deriv,tokens) for deriv in derivations]
                derivations.sort(lambda deriv:deriv[-1][1],reverse=True)
                
            if derivations:
                for idx,derivation in enumerate(derivations):
                    if idx < kbest:
                        r_tree = self.deriv2tree([action for action,prob in derivation])
                        r_tree.expand_unaries()
                        r_tree.add_gold_tags(tag_nodes)
                        if kbest > 1:
                            print(r_tree,derivation[-1][1],file=ostream)
                        else:
                            print(r_tree,file=ostream)
            else:
                print('(())',file=ostream,flush=True)

                
    @staticmethod
    def load_model(model_name):
        """
        Loads an RNNG parser from params at prefix model_name
        Args:
            model_name   (string): the prefix path for param files
        Returns:
            RNNGparser. An instance of RNNG ready to use.
        """
        parser = DiscoRNNGparser(model_name+'.conf')

        parser.lexicon      = SymbolLexicon.load(model_name+'.lex')
        parser.nonterminals = SymbolLexicon.load(model_name+'.nt')
        parser.tags         = SymbolLexicon.load(model_name+'.tag')
        
        parser.brown_file   = parser.brown_file+'.unk' #hack triggered by the 'code_lexicon' function
        parser.code_struct_actions()
        parser.allocate_conditional_params()
        parser.allocate_generative_params()
        parser.cond_model.populate(model_name+".cond.weights")
        parser.gen_model.populate(model_name+".gen.weights")
        return parser
                 
    def save_model(self,model_name):
        """
        Saves the model params using the prefix model_name.
        Args:
            model_name   (string): the prefix path for param files
        """       
        self.lexicon.save(model_name+'.lex')
        self.tags.save(model_name+'.tag')
        self.nonterminals.save(model_name+'.nt')
        self.cond_model.save(model_name+".cond.weights")
        self.gen_model.save(model_name+".gen.weights")

    def read_learning_params(self,configfile,conditional):

        config = configparser.ConfigParser()
        config.read(configfile)

        section = 'conditional' if conditional else 'generative'
        
        lr           = float(config[section]['learning_rate'])
        epochs       = int(config[section]['num_epochs'])
        dropout      = float(config[section]['dropout'])
        word_dropout = float(config[section]['word_dropout'])
        return lr,epochs,dropout,word_dropout

    
    def summary(self,train_bank_size,dev_bank_size,learning_rate,epochs,conditional):
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

        model_str   = 'discriminative' if conditional else 'generative'
        embed_size  = self.cond_word_embedding_size if conditional else  self.gen_word_embedding_size
        xstack_size = self.cond_stack_xsymbol_size  if conditional else  self.gen_stack_xsymbol_size
        mem_size    = self.cond_stack_memory_size   if conditional else  self.gen_stack_memory_size 

        summary_items = ['----------------------------',\
                          '* Training %s model *'%(model_str,),\
                          'Vocabulary   size   : %d'%(self.lexicon.size()),\
                          '# Nonterminals      : %d'%(self.nonterminals.size()),\
                          'Word embedding size : %d'%(embed_size),\
                          'Stack xinput size   : %d'%(xstack_size),\
                          'Stack memory size   : %d'%(mem_size),\
                          '# training trees    : %d'%(train_bank_size),\
                          '# validation trees  : %d'%(dev_bank_size),\
                          '# epochs            : %d'%(epochs),\
                          'Learning rate       : %.3f'%(learning_rate),\
                          'Dropout             : %.3f'%(self.dropout)]
        if conditional: 
            summary_items.extend(['POS embedding size  : %d'%(self.pos_embedding_size,),\
                                  'Hist embedding size : %d'%(self.history_xsymbol_size),\
                                  'Word dropout(alpha) : %.3f'%(self.word_dropout)])
        summary_items.append('----------------------------')
        return '\n'.join(summary_items)
                                  
    def estimate_params(self,train_treebank,train_tags,dev_treebank,dev_tags,modelname,config_file,conditional):
        """
        Estimates the parameters of a model from a treebank.
        """
        lr,epochs,dropout,word_dropout = self.read_learning_params(config_file,conditional)
        
        self.dropout      = dropout
        self.word_dropout = word_dropout
        #self.trainer      = dy.SimpleSGDTrainer(self.cond_model,learning_rate=lr) if conditional else dy.SimpleSGDTrainer(self.gen_model,learning_rate=lr)
        #experimental : add lr param if it works
        self.trainer      = dy.AdamTrainer(self.cond_model,alpha=lr) if conditional else dy.SimpleSGDTrainer(self.gen_model,learning_rate=lr)
        min_nll           = np.inf

        ntrain_sentences = len(train_treebank)
        ndev_sentences   = len(dev_treebank)

        print(self.summary(ntrain_sentences,ndev_sentences,lr,epochs,conditional))
        
        train_stats = RuntimeStats('NLL','lexNLL','N','lexN')
        valid_stats = RuntimeStats('NLL','lexNLL','N','lexN')
        
        for e in range(epochs):
            
            train_stats.push_row() 
            for idx,(tree,xtags) in enumerate(zip(train_treebank,train_tags)):
                train_stats += self.eval_sentence(tree,xtags,conditional=conditional,backprop=True)
                sys.stdout.write('\r===> processed %d training trees'%(idx+1))

            NLL,lex_NLL,N,lexN = train_stats.peek()            
            print('\n[Training]   Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL,np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)

            valid_stats.push_row()  
            for idx,(tree,xtags) in enumerate(zip(dev_treebank,dev_tags)):
                valid_stats += self.eval_sentence(tree,xtags,conditional=conditional,backprop=False)
 
            NLL,lex_NLL,N,lexN = valid_stats.peek()
            print('[Validation] Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL,np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)
            print()
            if NLL < min_nll:
                self.save_model(modelname+'/'+modelname) 
         
    def train_model(self,train_stream,dev_stream,modelname,config_file=None,conditional=True,generative=True):
        """
        Reads data and runs the learning process
        Args:
           train_stream  (stream): a stream open on a treebank file 
           dev_stream    (stream): a stream open on a treebank file 
        """
        #preprocessing
        train_treebank = [ ]
        train_words    = [ ]
        train_tags     = [ ]
        for line in train_stream:
            t      = DiscoTree.read_tree(line)
            tokens = t.pos_nodes()
            words  = [tok.children[0].label for tok in tokens]
            tags   = [tok.label             for tok in tokens]
            t.strip_tags()
            t.close_unaries()
            train_treebank.append(t)
            train_words.append(words)
            train_tags.append(tags)
        
        dev_treebank = [ ]
        dev_words    = [ ]
        dev_tags     = [ ]
        for line in dev_stream:
            t = DiscoTree.read_tree(line)
            tokens = t.pos_nodes()
            words  = [tok.children[0].label for tok in tokens]
            tags   = [tok.label             for tok in tokens]
            t.strip_tags()
            t.close_unaries()
            dev_treebank.append(t)
            dev_words.append(words)
            dev_tags.append(tags)
        
        self.code_lexicon(train_words,train_tags)
        self.code_nonterminals(train_treebank,dev_treebank)
        self.code_struct_actions()

        self.allocate_conditional_params( ) 
        self.allocate_generative_params( )   
        #Training 
        if conditional:
            self.estimate_params(train_treebank,train_tags,dev_treebank,dev_tags,modelname,config_file,True)
        if generative:
            self.estimate_params(train_treebank,train_tags,dev_treebank,dev_tags,modelname,config_file,False)

    def tests():
        t = DiscoTree.read_tree('(S (NP 0=John) (VP (VB 1=eats) (NP (DT 2=an) (NN 3=apple))) (PONCT 4=.))')
        print(t)
        wordlist = t.words()
        print(wordlist)
        print()

        p = DiscoRNNGparser()
        D,C = p.static_oracle(t,t,wordlist)
        print(D)
        print(p.deriv2tree(D))
        print()
    
        t = DiscoTree.read_tree('(S (VP (VB 0=is) (JJ 2=rich)) (NP 1=John) (PONCT 3=?))')
        print(t)
        wordlist = t.words()
        print(wordlist)
        print()
        
        p = DiscoRNNGparser()
        D,C = p.static_oracle(t,t,wordlist)
        print(D)
        print(p.deriv2tree(D)) 
        print() 
    
        t2 = DiscoTree.read_tree("(ROOT (SBARQ (SQ (VP (WHADVP (WRB 0=Why)) (VB 4=cross) (NP (DT 5=the) (NN 6=road))) (VBD 1=did) (NP (DT 2=the) (NN 3=chicken))) (PONCT 7=?)))")
        print(t2,'gap_degree',t2.gap_degree())
        wordlist = t2.words()
        print(wordlist)
        print() 

        D,C = p.static_oracle(t2,t2,wordlist)
        print(D)
        print(p.deriv2tree(D))
        print()
    
        t3 =  DiscoTree.read_tree('(S (X (A 0=a) (A 3=a))  (Y (B 1=b) (B 4=b)) (Z (C 2=c) (C 5=c)))')
        print(t3,'gap_degree',t3.gap_degree())
        wordlist = t3.words()
        print(wordlist)
        print()

        D,C = p.static_oracle(t3,t3,wordlist)
        print(D)
        print(p.deriv2tree(D))
        print() 

        #t4 = DiscoTree.read_tree('(ROOT (CS (S (CNP (NP (ART 0=Die) (MPN (NE 1=Rolling) (NE 2=Stones))) (KON 3=oder) (MPN (NE 4=Led) (NE 5=Zeppelin))) (VAFIN 6=haben) (NP (ADV 7=auch) (PIAT 8=keinen) (NE 9=Grammy))) ($, 10=,) (KON 11=und) (S (NP (PDS 12=die) (AP (PIAT 16=mehr) (NP (KOKOM 18=als) (PPER 19=ich)))) (VAFIN 13=hätten) (VP (PPER 14=ihn) (ADV 15=sicherlich) (VVPP 17=verdient)))) (D. 20=.) (D[ 21="))')
        t4 = DiscoTree.read_tree('(S (NP (PDS 0=die) (AP (PIAT 4=mehr) (NP (KOKOM 6=als) (PPER 7=ich)))) (VAFIN 1=hätten) (VP (PPER 2=ihn) (ADV 3=sicherlich) (VVPP 5=verdient)))')
        t4.close_unaries()
        wordlist = t4.words()
        print(wordlist)

        D,C = p.static_oracle(t4,t4,wordlist)
        print(D)
        print(p.deriv2tree(D))
        print()
            
                
if __name__ == '__main__':

    import shutil,os
    
    model_name      = ''
    config_file    = ''
    train_file     = ''
    dev_file       = ''
    pred_file      = '' #raw or mrg
    beam_size      = 32
    stats          = False
    discriminative = False
    generative     = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:d:p:m:b:c:sDGB:")
        for opt, arg in opts:
            if opt   in ['-t','--train']:
                train_file = arg
            elif opt in ['-d','--dev']:
                dev_file = arg
            elif opt in ['-p','--pred']:
                pred_file = arg
            elif opt in ['-c','--config']:
                config_file = arg
            elif opt in ['-m','--model']:
                model_name = arg
            elif opt in ['-G','--generative']:
                generative = True
            elif opt in ['-D','--discriminative']:
                discriminative = True
            elif opt in ['-B','--Beam-size']:
                beam_size  = int(arg)
            elif opt in ['-s','--stats']:
                stats = True
    except getopt.GetoptError:
        print('bad command line arguments.\naborting...') 

    if train_file and dev_file and model_name:
        try:
            os.mkdir(model_name)
            shutil.copyfile(config_file,model_name+'/'+model_name+'.conf')
        except:
            pass
        
        parser = DiscoRNNGparser(config_file=config_file)
        train_stream = open(train_file) 
        dev_stream   = open(dev_file)
        parser.train_model(train_stream,dev_stream,model_name,config_file=config_file,conditional=discriminative,generative=generative)
        train_stream.close()
        dev_stream.close()

    if model_name and pred_file:
        
        pred_stream = open(pred_file)
        out_stream  = open(model_name+'/'+model_name+'.pred.mrg','w')
        evalb_flag  = pred_file.endswith('mrg')

        parser = DiscoRNNGparser.load_model(model_name+'/'+model_name)
        parser.parse_corpus(pred_stream,ostream=out_stream,evalb_mode=evalb_flag,K=beam_size,kbest=1,conditional=discriminative,generative=generative)

        pred_stream.close()
        out_stream.close()

   


    
 
