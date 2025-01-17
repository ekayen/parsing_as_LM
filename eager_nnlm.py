#! /usr/bin/env python


import os
import os.path
import pickle
import time
import numpy as np
import pandas as pd
import dynet_config
dynet_config.set_gpu()
import dynet as dy

from math import log,exp
from random import shuffle
from numpy.random import choice,rand
from collections import Counter
from lm_utils import NNLMGenerator
from dataset_utils import DependencyTree,UDtreebank_reader
    
class StackNode(object):
    """
    This is a class for storing structured nodes in the parser's stack 
    """
    __slots__ = ['root','ilc','irc','starlc','rc_node']

    def __init__(self,root_idx):
        self.root   = root_idx
        self.ilc    = root_idx
        self.irc    = root_idx
        self.starlc = root_idx
        self.rc_node = None
                        
    def copy(self):
        other = StackNode(self.root)
        other.ilc     = self.ilc
        other.irc     = self.irc
        other.starlc  = self.starlc
        other.rc_node = self.rc_node
        return other
            
    def starrc(self):
        """
        Computes and returns the index of the righmost child of
        this node (reflexive transitive closure of RC(x))
        """
        if self.rc_node is None:
            return self.root
        else:
            return self.rc_node.starrc()

    def copy_left_arc(self):
        """
        Creates a copy of this node for when it governs a left arc
        """
        other = self.copy()
        other.starlc = min(self.starlc,other.starlc)
        other.ilc    = min(self.ilc,other.ilc) 
        return other
    
    def copy_right_arc(self,right_node):
        """
        Creates a copy of this node for when it governs a right arc
        @param right_node: the governed right StackNode.
        """
        other        = self.copy()
        other.irc     = max(self.irc,other.root)
        other.rc_node = right_node
        return other
        
    def __str__(self):
        return str(self.root)

class ArcEagerGenerativeParser:

    """
    An arc eager language model with local training.

    Designed to run on a GPU (at least for training). 
    """

    #actions
    LEFTARC  = "L"
    RIGHTARC = "R"
    GENERATE = "G"
    PUSH     = "P"
    REDUCE   = "RD"
    
    #end of sentence (TERMINATE ACTION)
    TERMINATE = "E"

    #Undefined and unknown symbols
    IOS_TOKEN     = "__IOS__"
    EOS_TOKEN     = "__EOS__"
    UNKNOWN_TOKEN = "__UNK__"

    def __init__(self,embedding_size=300,hidden_size=300,tied_embeddings=True,parser_class='basic'):
        """
        @param embedding_size: size of the embeddings
        @param hidden_size: size of the hidden layer
        @param tied_embeddings: uses weight tying for input and output embedding matrices
        @param parser_class {'basic','extended','star-extended'} controls the number of sensors
        """
        assert(parser_class in ['basic','extended','star-extended'])

        #sets default generic params
        self.model            = None
        self.stack_length     = 3
        self.parser_class     = parser_class
        if self.parser_class == 'basic':
            self.node_size        = 1                                #number of x-values per stack node
        elif self.parser_class == 'extended':
            self.node_size        = 3 
        elif self.parser_class == 'star-extended':
            self.node_size        = 5
            
        self.input_length     = (self.stack_length+1)*self.node_size # stack (+1 = focus node)
        self.tied             = tied_embeddings


        self.embedding_size   = embedding_size
        self.hidden_size      = hidden_size

        #lexical coding
        self.word_codes       = None  
        self.rev_action_codes = None
        self.lexicon_size     = 0 

        #structural coding
        self.actions = [self.leftarc,self.rightarc,self.push,self.reduce_config,self.terminate]
        actions = [ArcEagerGenerativeParser.LEFTARC,\
                   ArcEagerGenerativeParser.RIGHTARC,\
                   ArcEagerGenerativeParser.PUSH,\
                   ArcEagerGenerativeParser.REDUCE,\
                   ArcEagerGenerativeParser.TERMINATE]
                   #Generate action is implied
        self.rev_action_codes = actions                   
        self.actions_codes = dict([(s,idx) for (idx,s) in enumerate(actions)])
        self.actions_size  = len(actions) 

    @staticmethod
    def load_model(dirname):

        istream = open(os.path.join(dirname,'params.pkl'),'rb')
        params = pickle.load(istream)
        istream.close()
        
        g = ArcEagerGenerativeParser(embedding_size=params['embedding_size'],\
                                    hidden_size=params['hidden_size'],\
                                    tied_embeddings=params['tied'],\
                                    parser_class=params['parser_class'])

        if g.parser_class == 'basic':
            g.node_size        = 1                             
        elif g.parser_class == 'extended':
            g.node_size        = 3 
        elif g.parser_class == 'star-extended':
            g.node_size        = 5
        g.input_length         = (g.stack_length+1) * g.node_size 

        istream = open(os.path.join(dirname,'words.pkl'),'rb')
        g.word_codes = pickle.load(istream)
        istream.close()
        g.rev_word_codes = [0]*len(g.word_codes)
        for word,idx in g.word_codes.items():
             g.rev_word_codes[idx] = word
        g.lexicon_size = len(g.rev_word_codes)

        g.model = dy.ParameterCollection()
        g.hidden_weights   = g.model.add_parameters((g.hidden_size,g.embedding_size*g.input_length))
        g.action_weights   = g.model.add_parameters((g.actions_size,g.hidden_size))
        g.input_embeddings = g.model.add_parameters((g.lexicon_size,g.embedding_size))
        if not g.tied:
            g.output_embeddings = g.model.add_parameters((g.lexicon_size,g.hidden_size))
        g.model.populate(os.path.join(dirname,'model.prm'))
        return g

        
    def save_model(self,dirname,epoch= -1,learning_curve=None):
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        #select parameters to save
        params = {'parser_class':self.parser_class,\
                  'embedding_size':self.embedding_size,\
                  'hidden_size':self.hidden_size,\
                  'lexicon_size':self.lexicon_size,\
                  'tied':self.tied,\
                  'epoch':str(epoch)}

        ostream = open(os.path.join(dirname,'params.pkl'),'wb')
        pickle.dump(params,ostream)
        ostream.close()

        ostream = open(os.path.join(dirname,'words.pkl'),'wb')
        pickle.dump(self.word_codes,ostream)
        ostream.close()
    
        self.model.save(os.path.join(dirname,'model.prm')) 

        if learning_curve is not None:
            learning_curve.to_csv(os.path.join(dirname,'learning_curve.csv'),index=False)

    def __str__(self):
        s = ['Stack size        : %d'%(self.stack_length),\
            'Node size         : %d'%(self.node_size),\
            'Embedding size    : %d'%(self.embedding_size),\
            'Hidden layer size : %d'%(self.hidden_size),\
            'Actions size      : %d'%(self.actions_size),\
            'Lexicon size      : %d'%(self.lexicon_size),\
            'Tied Embeddings   : %r'%(self.tied)]
        return '\n'.join(s)


            
    #TRANSITION SYSTEM
    def init_configuration(self,tokens):
        """
        Generates the init configuration 
        """
        #init config: S, None, 1... n, empty arcs, score=0
        return ([StackNode(0)],None,tuple(range(1,len(tokens))),[],0.0)
        
    def push(self,configuration,local_score=0.0):
        """
        Performs the push action and returns a new configuration
        """
        S,F,B,A,prefix_score = configuration
        return (S + [F], None,B,A,prefix_score+local_score) 

    def leftarc(self,configuration,local_score=0.0):
        """
        Performs the left arc action and returns a new configuration
        """
        S,F,B,A,prefix_score = configuration
        i,j = S[-1].root,F.root
        return (S[:-1],F.copy_left_arc(),B,A + [(j,i)],prefix_score+local_score) 

    def rightarc(self,configuration,local_score=0.0):
        S,F,B,A,prefix_score = configuration
        i,j = S[-1].root,F.root
        S[-1] = S[-1].copy_right_arc(F)
        return (S+[F],None, B, A + [(i,j)],prefix_score+local_score) 

    def terminate(self,configuration,local_score=0.0):
        S,F,B,A,prefix_score = configuration
        return (S,F,B,A,prefix_score+local_score)
    
    def reduce_config(self,configuration,local_score=0.0):
        S,F,B,A,prefix_score = configuration
        return (S[:-1],F,B,A,prefix_score+local_score)
    
    def generate(self,configuration,local_score=0.0):
        """
        Pseudo-Generates the next word (for parsing)
        """
        S,F,B,A,prefix_score = configuration
        return (S,StackNode(B[0]),B[1:],A,prefix_score+local_score) 

    def static_oracle(self,configuration,dtree):
        """
        A default static oracle.
        @param configuration: a parser configuration
        @param dtree: a dependency tree object
        @return the action to execute given config and reference arcs
        """
        S,F,B,A,score  = configuration
        reference_arcs = dtree.edges
        all_words   = range(dtree.N())

        if F is None and B:
            return (ArcEagerGenerativeParser.GENERATE,dtree.tokens[B[0]])     
        if S and F:
            i,j = S[-1].root, F.root
            if (j,i) in reference_arcs:
                return ArcEagerGenerativeParser.LEFTARC
            if (i,j) in reference_arcs:
                return ArcEagerGenerativeParser.RIGHTARC
            
        if S and any ([(k,S[-1].root) in A for k in all_words]) \
             and all ([(S[-1].root,k) in A for k in all_words if (S[-1].root,k) in reference_arcs]):
                return ArcEagerGenerativeParser.REDUCE
        if not F is None:
            return ArcEagerGenerativeParser.PUSH
        return ArcEagerGenerativeParser.TERMINATE
        
    def static_oracle_derivation(self,dtree):
        """
        This generates a static oracle reference derivation from a sentence
        @param ref_parse: a DependencyTree object
        @return : the oracle derivation as a list of (Configuration,action) triples
        """
        sentence = dtree.tokens
        
        C = self.init_configuration(sentence)
        action = self.static_oracle(C,dtree)
        derivation = []
        
        while action != ArcEagerGenerativeParser.TERMINATE :
            
            derivation.append((C,action,sentence))
            if   action ==  ArcEagerGenerativeParser.PUSH:
                C = self.push(C)
            elif action == ArcEagerGenerativeParser.LEFTARC:
                C = self.leftarc(C)
            elif action == ArcEagerGenerativeParser.RIGHTARC:
                C = self.rightarc(C)
            elif action == ArcEagerGenerativeParser.REDUCE:
                C = self.reduce_config(C)
            else:
                action, w = action
                assert(action ==  ArcEagerGenerativeParser.GENERATE)
                C = self.generate(C)
                
            action = self.static_oracle(C,dtree)
            
        derivation.append((C,action,sentence))
        return derivation

    
    #PARSING
    def predict_next_best_action(self,config,prev_action,sentence):
        """
        Predicts the next best couple (configuration,action)
        @param config: the current configuration
        @param sentence: the sentence to parse
        @return a couple (next_config, action_taken)
        """
        S,F,B,A,prefix_score = config
        if F is None and len(B) > 0 : #lexical action
            unk_token = self.word_codes[ArcEagerGenerativeParser.UNKNOWN_TOKEN]
            next_word = self.word_codes.get(sentence[B[0]],unk_token)
            X = self.make_representation(config,None,sentence,structural=False)
            if self.tied:
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                embeddings = [dy.pick(E, xidx) for xidx in X]
                xdense     = dy.concatenate(embeddings)
                pred       = dy.pickneglogsoftmax(E * dy.tanh( W * xdense ),next_word)
                C = self.generate(config,local_score= -pred.value())
                action =  (ArcEagerGenerativeParser.GENERATE,sentence[B[0]])
                return (C,action)
            else:    
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                O = dy.parameter(self.output_embeddings)
                embeddings = [dy.pick(E, xidx) for xidx in X]
                xdense     = dy.concatenate(embeddings)
                pred       = dy.pickneglogsoftmax(O * dy.tanh( W * xdense ),next_word)
                C = self.generate(config,local_score= -pred.value())
                action = (ArcEagerGenerativeParser.GENERATE,sentence[B[0]])
                return (C,action)
        else:  #structural action
            X = self.make_representation(config,None,sentence,structural=True) 
            dy.renew_cg()
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.input_embeddings)
            A = dy.parameter(self.action_weights)
            embeddings = [dy.pick(E, xidx) for xidx in X]
            xdense     = dy.concatenate(embeddings)
            preds      = dy.softmax(A * dy.tanh( W * xdense )).npvalue()
            action_mask = self.mask_actions(config,prev_action,len(sentence))
            max_idx = np.argmax(preds*action_mask)
            score = log(preds[max_idx])
            C = self.actions[max_idx](config,local_score=score) #this just execs the predicted action..
            action = self.rev_action_codes[max_idx]
            return (C,action)

    def mask_actions(self,config,prev_action,N):
        """
        Return a boolean vector of dims = sizeof(action set)
        being a categorical mask for forbidden actions given some config.
        @param config: the current configuration
        @param prev_action: the previous action
        @param N: sentence length
        """
        S,F,B,A,prefix_score = config
        mask = np.ones(self.actions_size)

        if prev_action == ArcEagerGenerativeParser.TERMINATE:
            mask[self.actions_codes[ArcEagerGenerativeParser.RIGHTARC]] = 0
            mask[self.actions_codes[ArcEagerGenerativeParser.PUSH]]     = 0
            mask[self.actions_codes[ArcEagerGenerativeParser.LEFTARC]]  = 0
            mask[self.actions_codes[ArcEagerGenerativeParser.REDUCE]] = 0
            return mask
        if len(B) > 0:
            mask[self.actions_codes[ArcEagerGenerativeParser.TERMINATE]] = 0
        if F is None:
            mask[self.actions_codes[ArcEagerGenerativeParser.RIGHTARC]] = 0
            mask[self.actions_codes[ArcEagerGenerativeParser.PUSH]]     = 0
            mask[self.actions_codes[ArcEagerGenerativeParser.LEFTARC]]  = 0
        if len(S) == 0:
            mask[self.actions_codes[ArcEagerGenerativeParser.REDUCE]] = 0
            mask[self.actions_codes[ArcEagerGenerativeParser.RIGHTARC]] = 0
            mask[self.actions_codes[ArcEagerGenerativeParser.LEFTARC]] = 0
        else:
            s1_has_governor = any( (k, S[-1].root) in A for k in range(N))
            if s1_has_governor: #do not create cycles
                 mask[self.actions_codes[ArcEagerGenerativeParser.LEFTARC]] = 0
            else:               #do not omit words
                 mask[self.actions_codes[ArcEagerGenerativeParser.REDUCE]]  = 0
        return mask

    def K_best_actions(self,config,sentence):
        pass

    
    def greedy_parse(self,sentence):
        """
        Standard greedy parsing method for baseline.
        @param sentence : the sentence to parse (list of strings)
        @return a derivation (= a list of configurations,actions)
        """
        derivation = []
        C = self.init_configuration(sentence)
        C,action = self.predict_next_best_action(C,None,sentence)
        while action != ArcEagerGenerativeParser.TERMINATE:
            derivation.append((C,action,sentence))
            C,action = self.predict_next_best_action(C,action,sentence)
        return derivation


    def batch_greedy_parse(self,sentence_batch):
        """
        Parses greedily a batch of sentences.
        Shorter derivations are padded with TERMINATE actions at the end.
        @param sentence_batch : the sentences to parse (list of list of strings)
        @return a DependencyTree object
        """
        B = len(sentence_batch)
        derivation_batched = []
        config_batched = [self.init_configuration(sentence) for sentence in sentence_batch]
        config_batched,action_batched = self.batch_predict_next_best_action(config_batched,None,sentence_batch)
        while not all([action == ArcEagerGenerativeParser.TERMINATE for action in action_batched]):
            derivation_batched.append((config_batched,action_batched,sentence_batch))
            config_batched,action_batched = self.batch_predict_next_best_action(config_batched,action_batched,sentence_batch)
        return derivation_batched

    def batch_predict_next_best_action(self,config_batched,prev_action_batched,sentence_batch):
        """
        Predicts greedily the next transition for a batch of configs,
        actions leading to that config,and related sentences
        @param config_batched: a list of configurations
        @param prev_action_batched: a list of actions (or None if no prev actions)
        @param sentence_batch: a list of sentences
        @return a list of new configurations, a list of actions generating these new configs
        """
    
        B = len(config_batched)
        idxes = list(range(B))
        new_configs = [None] * B
        new_actions = [None] * B

        if prev_action_batched is None:
            prev_action_batched = [None]*B
                
        #(1) sort out the lexical and structural batches
        def is_lexical(config):
            S,F,B,A,prefix_score = config
            return F is None and len(B) > 0

        lexical_idxes    = [idx for idx in idxes if     is_lexical(config_batched[idx])]
        structural_idxes = [idx for idx in idxes if not is_lexical(config_batched[idx])]

        #(2) lexical predictions
        if len(lexical_idxes) > 0:

            def make_ref_lex_action(config,sentence):
                S,F,B,A,prefix_score = config
                return (ArcEagerGenerativeParser.GENERATE,sentence[B[0]])

            X = []
            Y = []
            for idx in lexical_idxes:
                x,y = self.make_representation(config_batched[idx],make_ref_lex_action(config_batched[idx],sentence_batch[idx]),sentence_batch[idx],structural=False)
                X.append(x)
                Y.append(y)

            Xt = zip(*X)    #transpose
        
            if self.tied:
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                embeddings = [dy.pick_batch(E, xcol) for xcol in Xt]
                xdense     = dy.concatenate(embeddings)
                preds      = dy.pickneglogsoftmax_batch(E * dy.tanh( W * xdense ),Y).npvalue()[0]
            else:
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                O = dy.parameter(self.output_embeddings)
                embeddings = [dy.pick_batch(E, xcol) for xcol in Xt]
                xdense     = dy.concatenate(embeddings)
                preds      = dy.pickneglogsoftmax_batch(O * dy.tanh( W * xdense ),Y).npvalue()[0]

            preds = np.atleast_1d(preds)
                
            for pred_score,idx in zip(preds,lexical_idxes): 
                new_configs[idx] = self.generate(config_batched[idx],local_score= -pred_score)# execs the actions  
                new_actions[idx] = (ArcEagerGenerativeParser.GENERATE,sentence_batch[idx][config_batched[idx][2][0]])

        #(3) structural predictions
        if len(structural_idxes) > 0 :
            action_masks = np.array([self.mask_actions(config_batched[idx],prev_action_batched[idx],len(sentence_batch[idx])) for idx in structural_idxes])
            X = [self.make_representation(config_batched[idx],None,sentence_batch[idx],structural=True) for idx in structural_idxes]
            Xt = zip(*X)    #transpose
            dy.renew_cg()
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.input_embeddings)
            A = dy.parameter(self.action_weights)
            embeddings = [dy.pick_batch(E, xcol) for xcol in Xt]
            xdense     = dy.concatenate(embeddings)
            preds      = dy.softmax(A * dy.tanh( W * xdense )).npvalue().transpose()

            max_idxes      = np.argmax(preds * action_masks,axis=1) 
            max_scores     = np.log(preds[np.arange(preds.shape[0]),max_idxes])
            for argmax_idx,max_score,idx in zip(max_idxes,max_scores,structural_idxes): 
                new_configs[idx] = self.actions[argmax_idx](config_batched[idx],local_score=max_score)  #execs the actions  
                new_actions[idx] = self.rev_action_codes[argmax_idx]
        return (new_configs, new_actions)

    def batch_eval_lm(self,treebank,batch_size=64):
        """
        Greedily batch parses a corpus and returns eval scores.
        """
        #(1) make batches
        idxes = list(range(len(treebank)))
        buckets = {}
        max_len = 0
        for idx in idxes:
            L = treebank[idx].N()
            max_len = max(L,max_len) 
            if L in buckets:
                buckets[L].append(idx)
            else:                
                buckets[L] = [idx]

        batches = []
        current_batch = []
        for sent_length in range(max_len):
            if sent_length in buckets:
                examples = buckets[sent_length]
                while len(examples)+len(current_batch) > batch_size:
                    split_idx = batch_size - len(current_batch)
                    current_batch.extend(examples[:split_idx])
                    batches.append(current_batch)
                    current_batch = []
                    examples = examples[split_idx:]
                current_batch.extend(examples)
        if len(current_batch) > 0:
            batches.append(current_batch)

        # (2) run eval
        Ntoks     = 0
        Nsents    = 0
        log_probs = 0
        uas       = 0
        for B in batches:
            sentences   = [treebank[idx].tokens for idx in B]
            derivations = self.batch_greedy_parse(sentences)
            
            for idx,deriv in zip(B,derivations):
                ref_tree  = treebank[idx]
                tokens    = ref_tree.tokens
                
                S,F,B,A,prefix_score = deriv[-1][0]
                pred_tree = DependencyTree(edges=list(A),tokens=sent)
                uas       +=  ref_tree.accuracy(pred_tree)
                log_probs +=  sum(self.predict_lex_probs(deriv))
                Ntoks     += len(tokens)
            Nsents += len(sentences)
        return (exp(-log_probs/Ntoks),uas/Nsents)

    
    def predict_lex_probs(self,derivation):
        """
        Computes the probability P(W1 ... Wn) of this sentence
        as the product Prod_{i=1}^n P(Wi | Wi-1, Wi-2 ... )
        @param : a parse derivation ( or a tree structured beam *TODO* ) as returned by a parser
        @return a list of P(Wi | ...) for each i as log probabilities
        """
        #TODO make it work for beam.            
        logprobs = []
        last_score = 0
        for C,action,sentence in derivation:
            if type(action) == tuple:
                _,word = action 
                S,F,B,A,prefix_score = C
                logprobs.append(prefix_score-last_score) #P(w_i+1|w_i) = P(w_i,w_i+1)/P(w_i) in log space
                last_score = prefix_score
        return  logprobs
    
    def parse_sentence(self,sentence,stats=False,kbest=1):
        """
        @param sentence : the sentence to parse (list of strings)
        @param stats: boolean: outputs a pandas dataframe for analysis rather than the dependency tree
        @return a DependencyTree object
        """
        assert(kbest >= 1)

        if kbest == 1:
            deriv = self.greedy_parse(sentence)
            S,F,B,A,prefix_score = deriv[-1][0]
            deptree = DependencyTree(edges=list(A),tokens=sentence)

            if not stats:
                return deptree

            governors   = [-1]*(len(sentence))
            for (g,d) in deptree.edges:
                governors[d] = g #temporary index shift (recovered by DF constructor)
            surprisals = [0.0] + [-lp for lp in self.predict_lex_probs(deriv)] #0 for init dummy token
            unk_words = [w not in self.word_codes for w in sentence]

            
            if len(surprisals) > len(sentence):
                print('truncate!')
                surprisals = surprisals[:len(sentence)]
            if len(sentence) > len(surprisals):
                surprisals.extend([0.0]*(len(sentence)-len(surprisals)))
            
            return pd.DataFrame({'token':sentence[1:],'surprisal':surprisals[1:],'governor':governors[1:],'unk_word':unk_words[1:]},index=range(1,len(sentence)))
        else:
            #todo...
            pass
    

    def eval_lm(self,treebank,uas=True,ppl=True,kbest=1):
        """
        @param treebank: an evaluation treebank
        @param uas: output unlabelled accurracy
        @param ppl: output perplexity
        """
        assert(kbest == 1)
        
        nll       = 0
        uas_val   = 0
        Ntoks     = 0 
        for sent in treebank:

            deriv = self.greedy_parse(sent.tokens)            
            if ppl:
                nll   -= sum(self.predict_lex_probs(deriv))
                Ntoks += len(sent.tokens)
            if uas:
                S,F,B,A,prefix_score = deriv[-1][0]
                pred_tree            = DependencyTree(edges=list(A),tokens=sent.tokens)
                uas_val             += sent.accurracy(pred_tree)
                
        if ppl and not uas:
            return exp(nll/Ntoks)
        if uas and not ppl:
            uas_val /=len(treebank)
            return uas_val
        if uas and ppl:
            ppl_val  = exp(nll/Ntoks)
            uas_val /= len(treebank)
            return (ppl_val,uas_val)


        
    #CODING & SCORING SYSTEM
    def code_symbols(self,treebank,lexicon_size=9998):
        """
        Codes lexicon (x-data) and the list of action (y-data)
        on integers.
        @param treebank    : the treebank where to extract the data from
        @param lexicon_size: caps the lexicon to some vocabulary size (default = mikolov size)
        """
        #lexical coding
        lexicon = [ArcEagerGenerativeParser.IOS_TOKEN,\
                   ArcEagerGenerativeParser.EOS_TOKEN,\
                   ArcEagerGenerativeParser.UNKNOWN_TOKEN]
        lex_counts = Counter()
        for dtree in treebank:
            lex_counts.update(dtree.tokens)
        lexicon = [w for w,c in lex_counts.most_common(9998-3)]+lexicon
        self.rev_word_codes = list(lexicon)
        self.lexicon_size = len(lexicon)
        self.word_codes = dict([(s,idx) for (idx,s) in enumerate(self.rev_word_codes)])
        

    def read_glove_embeddings(self,glove_filename):
        """
        Reads embeddings from a glove filename and returns an embedding
        matrix for the parser vocabulary.
        @param glove_filename: the file where to read embeddings from
        @return an embedding matrix that can initialize an Embedding layer
        """
        print('Reading embeddings from %s ...'%glove_filename)

        embedding_matrix = (rand(self.lexicon_size,self.embedding_size) - 0.5)/10.0 #uniform init [-0.05,0.05]

        istream = open(glove_filename)
        for line in istream:
            values = line.split()
            word = values[0]
            widx = self.word_codes.get(word)
            if widx != None:
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_matrix[widx] = coefs
        istream.close()
        print('done.')

        return embedding_matrix

    def pprint_configuration(self,config,sentence,verbose=False):
        """
        Pretty prints a configuration
        """
        S,F,B,A,score=config

        stack = ''
        if len(S) > 3:
            stack = '[...] '
        if len(S) > 2:
            stack += ' '+sentence[S[-2].root] if not verbose else '(%s %s %s)' %(sentence[S[-2].root],sentence[S[-2].ilc],sentence[S[-2].irc])
        if len(S) > 1:
            stack += ' '+sentence[S[-1].root] if not verbose else '(%s %s %s)' %(sentence[S[-1].root],sentence[S[-1].ilc],sentence[S[-1].irc])
        focus = '_'
        if F is not None:
            focus = sentence[F.root] if not verbose else '(%s %s %s)' %(sentence[F.root],sentence[F.ilc],sentence[F.irc])

        return '(%s,%s,_,_):%f'%(stack,focus,score)

    def make_representation(self,config,action,sentence,structural=True):
        """
        Turns a configuration into a couple of vectors (X,Y) and
        outputs the coded configuration as a tuple of index vectors.
        @param configuration: a parser configuration
        @param action: the ref action code (as a string) or None if the ref action is not known
        @param structural : bool, switch between structural action (True) and lexical action (False)
        @param sentence: a list of tokens (strings)
        @return a couple (X,Y) or just X if no action is given as param
        """
        S,F,B,A,score = config
        X  = [self.word_codes[ArcEagerGenerativeParser.IOS_TOKEN]] * self.input_length
        Ns = len(S)
        unk_token = self.word_codes[ArcEagerGenerativeParser.UNKNOWN_TOKEN]

        if type(action)==tuple:#lexical actions -> want the the word
            assert(not structural)
            a,word = action
            action = word
        
        if F is not None: X[0] = self.word_codes.get(sentence[F.root],unk_token)
        if Ns > 0 :       X[1] = self.word_codes.get(sentence[S[-1].root],unk_token)  
        if Ns > 1 :       X[2] = self.word_codes.get(sentence[S[-2].root],unk_token)  
        if Ns > 2 :       X[3] = self.word_codes.get(sentence[S[-3].root],unk_token)  

        if self.node_size > 1 :
            if F is not None :
                X[4] = self.word_codes.get(sentence[F.ilc],unk_token)      
                X[5] = self.word_codes.get(sentence[F.irc],unk_token)
            if Ns > 0 :   
                X[6] = self.word_codes.get(sentence[S[-1].ilc],unk_token)   
                X[7] = self.word_codes.get(sentence[S[-1].irc],unk_token)
            if Ns > 1 :
                X[8] = self.word_codes.get(sentence[S[-2].ilc],unk_token)  
                X[9] = self.word_codes.get(sentence[S[-2].irc],unk_token)
            if Ns > 2 :  
                X[10] = self.word_codes.get(sentence[S[-3].ilc],unk_token)  
                X[11] = self.word_codes.get(sentence[S[-3].irc],unk_token)
            
        if self.node_size > 3:
            if F is not None :
                X[12] = self.word_codes.get(sentence[F.starlc],unk_token)
                X[13] = self.word_codes.get(sentence[F.starrc()],unk_token)   
            if Ns > 0 :
                X[14] = self.word_codes.get(sentence[S[-1].starlc],unk_token)    
                X[15] = self.word_codes.get(sentence[S[-1].starrc()],unk_token)
            if Ns > 1 :
                X[16] = self.word_codes.get(sentence[S[-2].starlc],unk_token)   
                X[17] = self.word_codes.get(sentence[S[-2].starrc()],unk_token)
            if Ns > 2 :
                X[18] = self.word_codes.get(sentence[S[-3].starlc],unk_token)  
                X[19] = self.word_codes.get(sentence[S[-3].starrc()],unk_token)  
                
        if action is None:
            return X
        else:
            Y = self.actions_codes[action] if structural else self.word_codes.get(action,unk_token)        
            return (X,Y)

    def make_data_generators(self,treebank,batch_size):
        """
        This returns two data generators suitable for use with dynet.
        One for the lexical submodel and one for the structural submodel
        @param treebank: the treebank (list of sentences) to encode
        @param batch_size: the size of the batches yielded by the generators
        @return (lexical generator, structural generator) as NNLM generator objects
        """
        X_lex    = []
        Y_lex    = []
        X_struct = []
        Y_struct = []
       
        for dtree in treebank:
            Deriv = self.static_oracle_derivation(dtree)
            for (config,action,sentence) in Deriv:
                if type(action) == tuple: #lexical action
                    x,y    = self.make_representation(config,action,sentence,structural=False)
                    X_lex.append(x)
                    Y_lex.append(y)
                else:                     #structural action
                    x,y    = self.make_representation(config,action,sentence,structural=True)
                    X_struct.append(x)
                    Y_struct.append(y)
                    
        lex_generator    = NNLMGenerator(X_lex,Y_lex,batch_size)
        struct_generator = NNLMGenerator(X_struct,Y_struct,batch_size*2) # (!) generally 2 times as numerous than lex batches and easier to learn
        return ( lex_generator , struct_generator )
        
    def predict_logprobs(self,X,Y,structural=True,hidden_out=False):
        """
        Returns the log probabilities of the predictions for this model (batched version).

        @param X: the input indexes from which to predict (each xdatum is expected to be an iterable of integers) 
        @param Y: a list of references indexes for which to extract the prob
        @param structural: switches between structural and lexical logprob evaluation
        @param hidden_out: outputs an additional list of hidden dimension vectors
        @return the list of predicted logprobabilities for each of the provided ref y in Y
        """
        assert(len(X) == len(Y))
        assert(all(len(x) == self.input_length for x in X))

        if structural:
            dy.renew_cg()
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.input_embeddings)
            A = dy.parameter(self.action_weights)
            
            batched_X  = zip(*X) #transposes the X matrix
            embeddings = [dy.pick_batch(E, xcolumn) for xcolumn in batched_X]
            xdense     = dy.concatenate(embeddings)
            preds      = dy.pickneglogsoftmax_batch(A * dy.tanh( W * xdense ),Y).value()
            return [-ypred  for ypred in preds]

        else:#lexical
            if self.tied:
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                batched_X  = zip(*X) #transposes the X matrix
                embeddings = [dy.pick_batch(E, xcolumn) for xcolumn in batched_X]
                xdense     = dy.concatenate(embeddings)
                preds      = dy.pickneglogsoftmax_batch(E * dy.tanh( W * xdense ),Y).value()
                return [-ypred  for ypred in preds]
            else:
                dy.renew_cg()
                O = dy.parameter(self.output_embeddings)
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                batched_X  = zip(*X) #transposes the X matrix
                embeddings = [dy.pick_batch(E, xcolumn) for xcolumn in batched_X]
                xdense     = dy.concatenate(embeddings)
                preds      = dy.pickneglogsoftmax_batch(O * dy.tanh( W * xdense ),Y).value()
                return [-ypred  for ypred in preds]
    
    def static_train(self,\
                    train_treebank,\
                    validation_treebank,\
                    lr=0.001,\
                    hidden_dropout=0.01,\
                    batch_size=64,\
                    max_epochs=200,\
                    max_lexicon_size=9998,\
                    glove_file=None):
        """
        Locally trains a model with a static oracle and a multi-task standard feedforward NN.  
        @param train_treebank      : a list of dependency trees
        @param validation_treebank : a list of dependency trees
        @param lr                  : learning rate
        @param hidden_dropout      : dropout on hidden layer
        @param batch_size          : size of mini batches
        @param max_epochs          : max number of epochs
        @param max_lexicon_size    : max number of entries in the lexicon
        @param glove_file          : file where to find pre-trained word embeddings   
        """
        print("Encoding dataset from %d trees."%len(train_treebank))

        #(1) build dictionaries
        self.code_symbols(train_treebank,lexicon_size = max_lexicon_size)

        #(2) encode data sets
        lex_train_gen , struct_train_gen  = self.make_data_generators(train_treebank,batch_size)
        lex_dev_gen   , struct_dev_gen    = self.make_data_generators(validation_treebank,batch_size)
        
        print(self,flush=True)
        print("epochs %d\nstructural training examples  [N] = %d\nlexical training examples  [N] = %d\nBatch size = %d\nDropout = %f\nlearning rate = %f"%(max_epochs,struct_train_gen.N,lex_train_gen.N,batch_size,hidden_dropout,lr),flush=True)

        #(3) make network
        self.model = dy.ParameterCollection()
        self.hidden_weights   = self.model.add_parameters((self.hidden_size,self.embedding_size*self.input_length))
        self.action_weights   = self.model.add_parameters((self.actions_size,self.hidden_size))
        if glove_file is None:
            self.input_embeddings  = self.model.add_parameters((self.lexicon_size,self.embedding_size))
        else:
            self.input_embeddings  = self.model.parameters_from_numpy(self.read_glove_embeddings(glove_file))
        if not self.tied:
            self.output_embeddings = self.model.add_parameters((self.lexicon_size,self.hidden_size))

        #(4) fitting
        lex_gen       = lex_train_gen.next_batch()
        struct_gen    = struct_train_gen.next_batch()
        max_batches = max( lex_train_gen.get_num_batches(), struct_train_gen.get_num_batches() )
        print(lex_train_gen.get_num_batches(), struct_train_gen.get_num_batches(),flush=True)
        
        lex_valid_gen       = lex_dev_gen.next_batch()
        struct_valid_gen    = struct_dev_gen.next_batch()
        
        min_nll = float('inf')
        trainer = dy.AdamTrainer(self.model,alpha=lr)
        history_log = []
        for e in range(max_epochs):
            struct_loss,lex_loss = 0,0
            struct_N,lex_N       = 0,0
            start_t = time.time()
            for b in range(max_batches):
                #struct
                X_struct,Y_struct = next(struct_gen)
                #question of proportions : should struct and lex be evenly sampled or not (??):
                #here the parity oversamples approx twice the lexical actions
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                A = dy.parameter(self.action_weights)
                batched_X        = zip(*X_struct)  #transposes the X matrix                           
                lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                xdense           = dy.concatenate(lookups)
                ybatch_preds     = dy.pickneglogsoftmax_batch(A * dy.dropout(dy.tanh( W * xdense ),hidden_dropout),Y_struct)
                loss             = dy.sum_batches(ybatch_preds)
                struct_N         += len(Y_struct)
                struct_loss      += loss.value()
                loss.backward()
                trainer.update()
                #lex
                X_lex,Y_lex = next(lex_gen)
                if self.tied:
                    dy.renew_cg()
                    W = dy.parameter(self.hidden_weights)
                    E = dy.parameter(self.input_embeddings)
                    batched_X        = zip(*X_lex) #transposes the X matrix
                    lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                    xdense           = dy.concatenate(lookups)
                    ybatch_preds     = dy.pickneglogsoftmax_batch(E * dy.dropout(dy.tanh( W * xdense ),hidden_dropout),Y_lex)
                    loss             = dy.sum_batches(ybatch_preds)
                else:
                    dy.renew_cg()
                    W = dy.parameter(self.hidden_weights)
                    E = dy.parameter(self.input_embeddings)
                    O = dy.parameter(self.output_embeddings)
                    batched_X        = zip(*X_lex) #transposes the X matrix
                    lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                    xdense           = dy.concatenate(lookups)
                    ybatch_preds     = dy.pickneglogsoftmax_batch(O * dy.dropout(dy.tanh( W * xdense ),hidden_dropout),Y_lex)
                    loss             = dy.sum_batches(ybatch_preds)
                lex_N            += len(Y_lex)
                lex_loss         += loss.value()
                loss.backward()
                trainer.update()
            end_t = time.time()
            # (5) validation
            X_lex_valid,Y_lex_valid = lex_dev_gen.batch_all()
            lex_valid_nll           = -sum(self.predict_logprobs(X_lex_valid,Y_lex_valid,structural=False))
            
            X_struct_valid,Y_struct_valid = struct_dev_gen.batch_all()
            struct_valid_nll              = -sum(self.predict_logprobs(X_struct_valid,Y_struct_valid,structural=True))
            
            history_log.append((e,end_t-start_t,\
                                exp(lex_loss/lex_N),\
                                exp(struct_loss/struct_N),\
                                exp(lex_valid_nll/lex_dev_gen.N),\
                                exp(struct_valid_nll/struct_dev_gen.N),\
                                exp((lex_valid_nll+struct_valid_nll) /(struct_dev_gen.N+lex_dev_gen.N))))
            print('Epoch %d (%.2f sec.) TRAIN:: PPL_lex = %f, PPL_struct = %f / VALID:: PPL_lex = %f, PPL_struct = %f, PPL_all = %f'%tuple(history_log[-1]),flush=True)
            if  lex_valid_nll+struct_valid_nll < min_nll:
                df = pd.DataFrame(history_log,columns=['epoch','wall_time','ppl_lex_train','ppl_struct_train','ppl_lex_valid','ppl_struct_valid','ppl_all_valid'])
                self.save_model('best_model_dump',epoch = e, learning_curve=df)
            
        return pd.DataFrame(history_log,columns=['epoch','wall_time','ppl_lex_train','ppl_struct_train','ppl_lex_valid','ppl_struct_valid','ppl_all_valid'])

    # def generate_sentence(self,max_len=2000,lex_stats=False):
    #     """
    #     @param lex_stats: generate a table with word,log(prefix_prob),log(local_prob),num_actions
    #     """
    #     C = self.init_configuration()
    #     derivation = [C]
    #     action,score =  self.stochastic_oracle(C,is_first_action=True)
    #     stats = []
    #     while action != ArcEagerGenerator.TERMINATE and len(derivation) < max_len:
            
    #         if action == ArcEagerGenerator.LEFTARC:
    #             C = self.leftarc(C,score)
    #         elif action == ArcEagerGenerator.RIGHTARC:
    #             C = self.rightarc(C,score)
    #         elif action == ArcEagerGenerator.REDUCE:
    #             C = self.reduce_config(C,score)
    #         elif action == ArcEagerGenerator.PUSH:
    #             C = self.push(C,score)
    #         else:
    #             action, w = action
    #             assert(action == ArcEagerGenerator.GENERATE)
    #             C = self.generate(C,w,score)
    #             stats.append((w,C[4],log(score),len(derivation)))
    #         derivation.append(C)

    #         action,score = self.stochastic_oracle(C)
            
    #     if lex_stats:
    #         df = pd.DataFrame(stats,columns=['word','log(P(deriv_prefix))','log(P(local))','nActions'])
    #         return df
    #     else:
    #         return derivation

    
    # #CODING & SCORING
    # def stochastic_oracle(self,configuration,is_first_action=False):
    #     S,F,terminals,A,score = configuration
    #     X = np.array([self.make_representation(configuration)])
    #     Y = self.model.predict(X,batch_size=1)[0]
        
    #     def has_governor(node_idx,arc_list):
    #         """
    #         Checks if a node has a governor
    #         @param node_idx: the index of the node
    #         @arc_list: an iterable over arc tuples
    #         @return a boolean
    #         """
    #         return any(node_idx  == didx for (gidx,didx) in arc_list)
        
    #     if is_first_action:#otherwise predicts terminate with p=1.0 because init config is also the config right before calling terminate
    #         while True:
    #             action_code = choice(self.actions_size)#uniform draw
    #             action_score = 1.0/self.actions_size
    #             action = self.rev_action_codes[action_code]
    #             if type(action) == tuple:#this is a generate action
    #                 return (action,action_score)            
    #     if not S or F is None or S[-1].root == 0 or not has_governor(S[-1].root,A):
    #         Y[self.actions_codes[ArcEagerGenerator.LEFTARC]] = 0.0
    #     if not S or F is None:
    #         Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]] = 0.0
    #     if F is None or not S or not has_governor(S[-1].root,A):
    #         Y[self.actions_codes[ArcEagerGenerator.REDUCE]] = 0.0
    #     if F is None:
    #         Y[self.actions_codes[ArcEagerGenerator.PUSH]] = 0.0
    #     if F is not None:
    #         la = Y[self.actions_codes[ArcEagerGenerator.LEFTARC]]
    #         ra = Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]]
    #         r  = Y[self.actions_codes[ArcEagerGenerator.REDUCE]]
    #         p  = Y[self.actions_codes[ArcEagerGenerator.PUSH]]
    #         t  = Y[self.actions_codes[ArcEagerGenerator.TERMINATE]]
            
    #         Y = np.zeros(self.actions_size)
            
    #         Y[self.actions_codes[ArcEagerGenerator.LEFTARC]]   = la
    #         Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]]  = ra
    #         Y[self.actions_codes[ArcEagerGenerator.REDUCE]]    = r
    #         Y[self.actions_codes[ArcEagerGenerator.PUSH]]      = p
    #         Y[self.actions_codes[ArcEagerGenerator.TERMINATE]] = t
            
    #     Z = Y.sum()
    #     if Z == 0.0:#no action possible, trapped in dead-end, abort.
    #         return (ArcEagerGenerator.TERMINATE,np.finfo(float).eps)
            
    #     Y /= Z
    #     action_code = choice(self.actions_size,p=Y)
    #     action_score = Y[action_code]
    #     action = self.rev_action_codes[action_code]

    #     #print distribution:
    #     #print ('kbest')
    #     #kbest = sorted([(p,idx) for (idx,p) in enumerate(Y)],reverse=True)[:20]
    #     #for p,idx in kbest:
    #     #    print(idx,self.rev_action_codes[idx],p)
                
    #     return (action,action_score)


if __name__ == '__main__':
    
    train_treebank = UDtreebank_reader('ptb/ptb_deps.train',tokens_only=False)
    dev_treebank   = UDtreebank_reader('ptb/ptb_deps.dev',tokens_only=False)
    
    eagerp = ArcEagerGenerativeParser(tied_embeddings=True,parser_class='basic')
    eagerp.static_train(train_treebank[:20],dev_treebank[:20],lr=0.001,hidden_dropout=0.2,batch_size=64,max_epochs=5,glove_file='glove/glove.6B.300d.txt')
    #print('PPL = %s ; UAS = %f'%eagerp.eval_lm(train_treebank,uas=True,ppl=True))
    #print('PPL = %s ; UAS = %f'%eagerp.eval_lm(dev_treebank,uas=True,ppl=True))
    #eagerp.save_model('final_model')
    #eagerp = ArcEagerGenerativeParser.load_model('final_model')
    print('PPL = %s ; UAS = %f'%eagerp.eval_lm(dev_treebank[:20],uas=True,ppl=True))    
    print('PPL = %s ; UAS = %f'%eagerp.batch_eval_lm(dev_treebank[:20]))

    #for s in dev_treebank[:15]:
    #   print(eagerp.parse_sentence(s.tokens,stats=True,kbest=1))        
