import os
import json
from code.PRETRIE import PrefixTrie
from code.CONTRIE import ContainTrie


def read_all(filename='./data/clean/train.txt'):
    lines = open(filename, 'r').readlines()
    lines = [x.strip('\n') for x in lines]
    return lines


def sort_list_by_second_element(lst):
    """
    Sorts a list according to the second element in every element in the list.
    
    Args:
    lst: list of tuples, where each tuple contains at least two elements
    
    Returns:
    A sorted list according to the second element in every element in the list.
    """
    return sorted(lst, key=lambda x: x[1], reverse=True)

def setup_models(pre_pt_filename='./models/prefix.json', cont_pt_filename='./models/contain.json'):
    if os.path.exists(pre_pt_filename) and os.path.exists(cont_pt_filename):
        print('模型加载中...')
        prefix_model = PrefixTrie()
        prefix_model = PrefixTrie.load(pre_pt_filename)
        
        contain_model = ContainTrie()
        contain_model = ContainTrie.load(cont_pt_filename)
        
    else:
        print('模型训练中...')
        lines = read_all()
    
        if not os.path.exists(pre_pt_filename):
            prefix_model = setup_prefix(lines)
            prefix_model.save(pre_pt_filename)
        if not os.path.exists(cont_pt_filename):
            contain_model = setup_contain(lines)
            contain_model.save(cont_pt_filename)

    print('加载完成.')
    return prefix_model, contain_model
        
def setup_prefix(lines):
    trie = PrefixTrie()
    # lines = read_all()
    
    for word in lines:
        trie.insert(word)
    return trie

def setup_contain(lines):
    trie = ContainTrie()
    # lines = read_all()
    
    for word in lines:
        trie.insert(word)
    return trie
