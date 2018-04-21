#!/usr/bin/env python


import numpy as np
import pandas as pd

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
    def __init__(self,filename,wordlist):
        """
        @param filename: file where to save the data
        @param wordlist: the list of word entries known to the parser
        """
        self.filename = filename
        self.vocabulary = set([])
        self.global_log = []
        self.next_sentence([])

    def set_known_vocabulary(self,wordlist):
        """
        @param wordlist: list of known tokens
        """
        self.vocabulary = set(wordlist)

    def log_beam_element(self,beam_element):
        """
        This tracks all the stats of interest from elements found in
        the lexical beam at some time step.
        
        @param beam_element: a beam element generated by the parser
        """
        _,_,_,_,_,prefix_logprob = beam_element.config

        #backtrack to prev lexical item config
        current  = beam_element.prev_item.prev_item
        if current == None: #start of sentence, first action is necessarily shift followed by word emission
            self.step_aggregate    += 2
            self.logprob_aggregate  = np.logaddexp(self.logprob_aggregate,prefix_logprob)
            self.num_configs       += 1
        else:               #regular case
            target_elt = None
            num_steps = 2
            while current.incoming_action != RNNGparser.SHIFT and current.prev_item != None:
                target_elt = current
                current = current.prev_item
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
        token = self.tokens[self.idx]
        is_unknown = not (token in self.vocabulary)
        surprisal = self.logprob_aggregate/np.log(2) #change log to base 2 for surprisal
        self.sent_log.append( (token,is_unknown,surprisal,self.step_aggregate,self.num_configs) )
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
        flat_dataset = [elt for elt in sent for sent in self.global_log]
        df = pd.DataFrame.from_records(self.ppl_dataset,columns=['token','is_unknown','surprisal','nactions','beam_size'])
        df.to_csv(self.filename)
