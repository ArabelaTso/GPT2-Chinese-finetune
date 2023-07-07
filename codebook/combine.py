import os
import sys
import json
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_word = False
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def save(self, file_path):
        data = self._serialize(self.root)
        with open(file_path, 'w') as f:
            json.dump(data, f)
            
    def _serialize(self, node):
        data = {
            'is_end_word': node.is_end_word,
            'count': node.count,
            'children': {}
        }
        for char, child_node in node.children.items():
            data['children'][char] = self._serialize(child_node)
        return data

    def _deserialize(self, data):
        node = TrieNode()
        node.is_end_word = data['is_end_word']
        node.count = data['count']
        for char, child_data in data['children'].items():
            node.children[char] = self._deserialize(child_data)
        return node
    

class PrefixTrie(Trie):
    def __init__(self):
        super().__init__()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_end_word = True
        node.count += 1

    def search(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        results = self._dfs(node, prefix)
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _dfs(self, node, prefix):
        results = []
        if node.is_end_word:
            results.append((prefix, node.count))
        for char, child in node.children.items():
            results.extend(self._dfs(child, prefix + char))
        return results
    
    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        trie = PrefixTrie()
        trie.root = trie._deserialize(data)
        return trie
    
    
class ContainTrie(Trie):
    def __init__(self):
        super().__init__()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
        node.count += 1

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        trie = ContainTrie()
        trie.root = trie._deserialize(data)
        return trie
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_word

    
    def contains(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def get_strings_with_word(self, word, node=None, prefix=''):
        if node is None:
            node = self.root
        results = []
        if node.is_end_word and word in prefix:
            results.append((prefix, node.count))
        for char in node.children:
            child_node = node.children[char]
            if self.contains(word):
                results.extend(self.get_strings_with_word(word=word, node=child_node, prefix=prefix + char))
        return sorted(results, key=lambda x: x[1], reverse=True)[:5]


def read_all():
    lines = open('../data/clean/train_all_new.txt', 'r').readlines()
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

def setup_models(pre_pt_filename='prefix.json', cont_pt_filename='contain.json'):
    if os.path.exists(pre_pt_filename) and os.path.exists(cont_pt_filename):
        print('Loading models...')
        prefix_model = PrefixTrie()
        prefix_model = PrefixTrie.load(pre_pt_filename)
        
        contain_model = ContainTrie()
        contain_model = ContainTrie.load(cont_pt_filename)
        
    else:
        print('Preparing models...')
        lines = read_all()
    
        if not os.path.exists(pre_pt_filename):
            prefix_model = setup_prefix(lines)
            prefix_model.save(pre_pt_filename)
        if not os.path.exists(cont_pt_filename):
            contain_model = setup_contain(lines)
            contain_model.save(cont_pt_filename)

    print('Models loaded.')
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

def get_replies_from_two_models(prefix_model, contain_model, prompt:str):
    try:
        results = []
        prompt = prompt.strip()
        complete = prefix_model.search(prompt)
        if len(complete) == 0:
            results.append('Cannot complete the sentence.')
        else:
            results.append(complete[0][0])
        
        imagines = []
        imagines.extend(contain_model.get_strings_with_word(word=prompt))
        if len(imagines) == 0:
            results.append('Cannot match any sentences.')
        else:
            for x in imagines[:min(len(imagines), 5)]:
                results.append(x[0])
        
        return results
    except Exception as e:
        print(e)
        return ['404', '404']


if __name__ == '__main__':
    pre_pt_filename = './models/prefix.json'
    cont_pt_filename = './models/contain.json'
    
    prefix_model, contain_model = setup_models(pre_pt_filename, cont_pt_filename)
    
    # for loop input
    while True:
        input_str = input("Input:")
        
        # results = get_replies_from_two_models(prefix_model, contain_model, input_str)
        # print(results)
        
        # Completion
        complete_results = prefix_model.search(input_str)
        if len(complete_results) == 0:
            complete_results = 'Cannot complete the sentence.'
            # print('Cannot complete the sentence.\n')
        else:
            complete_results = complete_results[0][0]
            # print("Complete: {}".format(complete_results))
            # print()
        
        # Contain
        contain_results = []
        contain_results.extend(contain_model.get_strings_with_word(word=input_str))
        
        if len(contain_results) == 0:
            contain_results = [('Cannot match any sentences.', 0)]
            # print('Cannot match any sentences.\n')
        else:
            contain_results = contain_results[:min(len(contain_results), 5)]
            # print("Imaginations: \n- {}".format('\n- '.join([x[0] for x in contain_results[:min(len(contain_results), 5)]])))
            # print()
            
        
        print("补全：{}\n联想：\n- {}\n\n".format(complete_results, '\n- '.join([x[0] for x in contain_results])))
        

