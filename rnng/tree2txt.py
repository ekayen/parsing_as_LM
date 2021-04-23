from constree import *
import os


split = 'dev'
tree_file = f'data/{split}.trees'
txt_file = f'data/{split}.txt'

with open(tree_file,'r') as f:
    lines = [l.strip() for l in f.readlines()]

with open(txt_file,'w') as f:
    for line in lines:
        toks = ConsTree.read_tree(line).tokens()
        f.write(' '.join(toks))
        f.write('\n')

    
