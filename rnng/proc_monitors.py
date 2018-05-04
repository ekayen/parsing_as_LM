#!/usr/bin/env python


import numpy as np
import pandas as pd
import rnng
from constree import *

"""
This module implements monitoring tools for tracking measures of
interest for cognitive modeling
"""
class AbstractTracker(object):
    """
    This specifies the abstract interface for tracker objects
    """
    def log_beam_element(self,beam_element):
        pass
    def set_known_vocabulary(self,voc):
        pass
    def next_word(self):
        pass
    def next_sentence(self,tokens):
        pass
    def save_table(self):
        pass
    
class DefaultTracker(AbstractTracker):
    """
    That's the default tracker. It does nothing exceptional but it defines the
    minimal interface and provides an example for getting measures
    from a parser. Meant to be subclassed.   
    """
    def __init__(self,filename):
        """
        @param filename: file where to save the data
        """
        self.filename = filename
        self.vocabulary = set([])
        self.global_log = []
        self.sent_log   = []
        self.next_sentence([])
        self.LL         = 0 
        self.N          = 0

    def get_corpus_NLL(self,ppl=False):
        """
        Returns the NLL of the corpus parsed so far.
        @param ppl: returns the perplexity instead
        """
        if ppl:
            return np.exp(-self.LL/self.N)
        return -self.LL
                
    def set_known_vocabulary(self,wordlist):
        """
        @param wordlist: list of known tokens
        """
        if not self.vocabulary:
            self.vocabulary = set(wordlist)

    def log_beam_element(self,beam_element):
        """
        This tracks all the stats of interest from elements found in
        the lexical beam at some time step.
        
        @param beam_element: a beam element generated by the parser
        """
        _,_,_,_,_,prefix_logprob = beam_element.config

        #backtrack to prev lexical item config
        current  = beam_element.prev_element.prev_element
        if current.incoming_action == 'init': #start of sentence, first action is necessarily shift followed by word emission
            self.step_aggregate    += 2
            self.logprob_aggregate  = np.logaddexp(self.logprob_aggregate,prefix_logprob)
            self.num_configs       += 1
        else:               #regular case
            target_elt = None
            num_steps = 2
            while current.incoming_action != rnng.RNNGparser.SHIFT and current.prev_element != None:
                target_elt = current
                current = current.prev_element
                num_steps+=1
            num_steps -= 1
            _,_,_,_,_,prev_prefix_logprob = target_elt.config
            self.step_aggregate    += num_steps
            self.logprob_aggregate  = np.logaddexp(self.logprob_aggregate,prefix_logprob-prev_prefix_logprob)
            self.num_configs       += 1
        
    def next_word(self):
        """
        Moves to the next word
        """
        if self.idx < len(self.tokens):
            token = self.tokens[self.idx]
            is_unknown = not (token in self.vocabulary)
            self.LL   += self.logprob_aggregate
            self.N    += 1
            surprisal = self.logprob_aggregate/np.log(2) #change log to base 2 for surprisal
            self.sent_log.append( (token,is_unknown,surprisal,self.step_aggregate/self.num_configs) )
            self.logprob_aggregate = 0
            self.step_aggregate    = 0
            self.num_configs       = 0
            self.idx += 1
        
    def next_sentence(self,tokens):
        """
        Moves to the next sentence
        @param tokens: a list of strings, the tokens from the sentence
        """
        if self.sent_log:
            self.global_log.append(self.sent_log)
            
        self.sent_log = []
        self.tokens = tokens
        self.idx = 0
        self.logprob_aggregate = 0
        self.step_aggregate    = 0
        self.num_configs       = 0
    
    def save_table(self):
        """
        Dumps the current state of the stats to a table without
        explicitly marking sentence boundaries.
        """
        if self.sent_log:#flush
            self.next_sentence([])
        
        flat_dataset = [elt for sent in self.global_log for elt in sent]
        #df = pd.DataFrame.from_records(flat_dataset,columns=['token','is_unknown','surprisal','mean_actions'])
        #df.to_csv(self.filename)
        ostream = open(self.filename,'w')
        print(','.join(['token','is_unknown','surprisal','mean_actions']),file=ostream)
        for line in flat_dataset:
            print(','.join([str(elt) for elt in line]),file=ostream)
        ostream.close()

####################################
#BEAM & SEARCH EXPLORATION FUNCTIONS 

def beam_search_debug(ref_tree,all_beam_size,lex_beam_size,kbest,tracker,WORD_BREAK_ACTION):
    """
    Parses a tree and dumps beam & search related displays.
    """
    ConsTree.strip_tags(ref_tree)
    tokens = ref_tree.tokens()
    results= p.beam_parse(tokens,all_beam_size=struct_beam,lex_beam_size=lex_beam,kbest=kbest,tracker=dtracker,get_derivation=True)
    for elt in results:
        if elt:
            deriv,_ = elt
            pred_tree = RNNGparser.derivation2tree(deriv,tokens)
            pred_tree.expand_unaries() 
            print("%s %f"%(str(pred_tree),t.compare(pred_tree)[2]),flush=True)
                    
    #Compares the best parse derivation with the reference annotation
    ConsTree.close_unaries(ref_tree)
    ref,rprobs = p.eval_sentence(ref_tree,get_derivation=True)
    pred_deriv,pprobs = results[0]

    i = 0
    for idx, elt in enumerate(pred_deriv):
        if type(elt) == int:
            pred_deriv[idx] = tokens[i]
            i += 1
    ref_sync  = word_sync_derivation(derivation,rprobs,WORD_BREAK_ACTION)
    pred_sync = word_sync_derivation(pred_deriv,pprobs,WORD_BREAK_ACTION)
    compare_derivations(ref_sync,pred_sync)
    
def word_sync_derivation(derivation,prob_sequence, WORD_BREAK_ACTION):
    """
    This function a derivation as a list of subderivations wordwise aligning words and
    probs.
    @param derivation: a derivation
    @param prob_sequence: a list of prefix probs same size as
    derivation
    @param WORD_BREAK_ACTION: the action that indicates the word boundary
    @return a list of subderivations
    """
    assert(len(derivation) == len(prob_sequence))
    start_idx = 0
    result = [] 
    for idx, deriv, prob in zip(range(len(derivation)),derivation,prob_sequence):
        if deriv == WORD_BREAK_ACTION:
            result.append(derivation[start_idx:idx+2])
            start_idx = idx+1
    result.append(derivation[start_idx:])


def compare_derivations(wsync_deriv_A,wsync_deriv_B,margin=40):
    """
    Prints two derivations word-wise aligned.
    @param : wsync_deriv_A = output of word_sync_derivation func
    @param : wsync_deriv_B = output of word_sync_derivation func
    """
    assert(len(wsync_deriv_A) == len(wsync_deriv_B))
    for X,Y in zip(wsync_deriv_A,wsync_deriv_B):

        line  = ', '.join(['%s:%f'%(d,p) for (d,p) in X])
        line += ' '*max(0,margin-len(line))
        line +=  ', '.join(['%s:%f'%(d,p) for (d,p) in Y])
        print(line)




        
