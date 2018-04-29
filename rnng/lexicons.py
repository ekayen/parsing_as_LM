#! /usr/bin/env python
from collections import Counter
import json
import numpy as np

"""
These are various lexer style utilities for coding strings to integers
and vice versa.
"""
class SymbolLexicon:
    """
    This class manages encoding of a lexicon as a set of finite size, including unknown words
    management. It provides services for mapping tokens to integer
    indexes and vice-versa.
    """
    def __init__(self,wordlist,unk_word=None,special_tokens=[ ],count_threshold=-1,max_lex_size=100000000):
        """
        @param wordlist       : a list of strings or a collections.Counter
        @param unk_word       : a token string for unknown words
        @param special tokens : a list of reserved tokens such as <start> or <end> symbols etc that are added to the lexicon 
        @param count_threshold: words are part of the lexicon if their counts is > threshold
        @param max_lex_size   : max number of elements in the lexicon
        """
        counts       = wordlist if isinstance(wordlist,Counter) else Counter(wordlist)
        lexlist      = [ word for word, c in counts.most_common(max_lex_size) if c > count_threshold ]

        if unk_word:
            lexlist.append(unk_word)
        self.UNK     = unk_word

        lexlist.extend(special_tokens)
        self.words2i = dict([ (w,idx) for idx,w in enumerate(lexlist)])
        self.i2words = lexlist
        #ordered list of symbols
        self.symlist = list(self.words2i.keys())

    def get_UNK_ID(self):
        """
        @return the integer index of the unknown word
        """
        if self.UNK:
            return self.words2i[self.UNK]
        else:
            return None
        
    def __str__(self):
        return ' '.join(self.words2i.keys())

    def __contains__(self,item):
        """
        Overrides the in operator
        @param item a word we want to know if it is known to the
        lexicon or not
        @return a boolean
        """
        return item in self.words2i
            
    def size(self):
        """
        @return the vocabulary size
        """
        return len(self.i2words)
        
    def normal_wordform(self,token):
        """
        @param token: a string
        @return a string, the token or the UNK word code
        """
        return token if token in self.words2i else self.UNK

    def index(self,token):
        """
        @param token: a string
        @return the index of the word in this lexicon
        """
        return self.words2i[ self.normal_wordform(token) ]
        
    def wordform(self,idx):
        """
        @param idx: the index of a token
        @return a string, the token for this index
        """
        return self.i2words[ idx ]
    
    def save(self,modelname):
        """
        @param modelname : the name of the model to save
        """
        ostream = open(modelname+'.json','w')
        ostream.write(json.dumps({'UNK':self.UNK ,'lex':self.i2words}))
        ostream.close()

    @staticmethod
    def load(modelname):
        """
        @param modelname : the name of the model to load
        @return a SymbolLexicon object
        """
        istream = open(modelname+'.json')
        struct = json.loads(istream.read())
        istream.close()
        
        UNK     = struct['UNK']
        lexlist = struct['lex']
        
        lex = SymbolLexicon([])
        lex.words2i = dict([ (w,idx)    for idx,w in enumerate(lexlist)])
        lex.i2words = lexlist
        lex.UNK     = UNK
        return lex

     
class BrownLexicon:
    """
    This class manages brown cluster as generated by P. Liang's
    clustering package. By hypothesis the set of wordforms is
    partitioned by the clusters : no wordform can belong to more than
    one cluster.
    """
    def __init__(self,w2cls,word_counts):
        """
        @param w2cls: a dict wordform          -> cluster IDs (IDs as integers)
        @param word_counts   : a Counter wordform -> counts
        #@param freq_threshold: max number of wordforms to include in the dictionary
        """
        self.w2cls = w2cls
        self.word_counts = word_counts
        self.UNK_ID = 0 #cluster ID for unknown token
        
        #computes the marginal counts of the clusters
        self.cls_counts = { }                             #raw counts of the clusters in the corpus
        for word,clust in self.w2cls.items():
            C = self.index(word)
            self.cls_counts[C] = self.cls_counts.get(C,0) + self.word_counts[word]


            
    def display_summary(self):
        return """Using Brown Clusters with %d clusters and a lexicon of %d word forms"""%(len(self.cls_counts),len(self.w2cls))
        
    def __str__(self):
        return '\n'.join( ['P(%s|%d) = %f'%(w,C,self.word_emission_prob(w,logprob=False)) for w,C in self.w2cls.items()])
        
    def size(self):
        """
        Returns the number of clusters.
        """
        return len(self.cls_counts)
          
    def index(self,wordform):
        """
        Returns the integer index of the cluster to which this word belongs or a default
        value if this word is unknown to the clustering
        @param wordform: a string
        @return an integer index of the cluster
        """
        return self.w2cls.get(wordform,self.UNK_ID)

    def word_emission_prob(self,wordform,logprob=True):
        """
        Returns P(w|C) if the word is known to the lexicon.
        Otherwise returns P(w=UNK|Unk cluster), that is 1.0. 
        
        @return P(w|C) that is the probability that this word is
        generated by its cluster C (which is automatically retrieved)
        
        @param wordform: the w of P(w|C)
        @param logprob:
        """
        C = self.index(wordform)
        if C == self.UNK_ID:
            return 0 if logprob else 1
        else:
            N = self.cls_counts[C]
            w = self.word_counts[wordform]
            p = float(w)/float(N)
            return np.log(p) if logprob else p
        
    def save_clusters(self,filename):
        """
        Saves the clusters in a json format
        """
        jfile = open(filename+'.json','w')
        jfile.write(json.dumps({'word_counts':self.word_counts,\
                                'w2cls':self.w2cls,\
                                'UNK_ID':self.UNK_ID}))
        jfile.close()

    @staticmethod
    def load_clusters(filename):
        """
        Loads the clusters from a json format
        """
        struct = json.loads(open(filename+'.json').read())
        blex =  BrownLexicon(struct['w2cls'],struct['word_counts'],freq_threshold=0)
        blex.UNK_ID = struct['UNK_ID']
        return blex
    
    @staticmethod
    def read_clusters(cls_filename,freq_thresh=-1,UNK_SYMBOL='<UNK>'):
        """
        Builds a BrownLexicon object from raw path files.
        Words with count = or less than fre_thresh are discarded.
        
        @param cls_filename: a path file produced by P. Liang Package
        @param freq_thresh: a count threshold above which words are included in the lexicon.
        @return a BrownLexicon object
        """
        istream = open(cls_filename)
        clsIDs  = set([])
        word_counts = {}
        w2cls = {}
        for line in istream:
            ID, word, counts = line.split()
            if int(counts) > freq_thresh:
                clsIDs.add(ID)
                w2cls[word] = ID
                word_counts[word] = int(counts)
        istream.close()
        
        #now remap clsIDs to continuous range of integers
        clsIDs = dict( [(C,idx+1) for (idx, C) in enumerate(clsIDs) ]) #UNK symbol has cluster ID 0 
        w2cls  = dict( [(w,clsIDs[ID]) for (w,ID) in w2cls.items()] )
        w2cls[UNK_SYMBOL]       = 0                                                 #UNK symbol has cluster ID 0 
        word_counts[UNK_SYMBOL] = 1
        return BrownLexicon(w2cls,word_counts)
        

if __name__ == '__main__':
    symlex = SymbolLexicon(["A"]*3+['B']*4+['C'],count_threshold=1)
    print(symlex)
    print(symlex.index('A'),symlex.index('B'),symlex.index('C'),symlex.index('D'))
    print(symlex.wordform(0),symlex.wordform(1),symlex.wordform(2))
    
    blex = BrownLexicon.read_clusters("/Users/bcrabbe/parsing_as_LM/rnng/cls_example.txt",freq_thresh=1)
    print(blex)
