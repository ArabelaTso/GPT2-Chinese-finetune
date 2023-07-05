import os
import json
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.end_of_word = False
        self.count = 0

class ContainTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True
        node.count += 1

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.end_of_word

    def save(self, file_path):
        data = self._serialize(self.root)
        with open(file_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        trie = ContainTrie()
        trie.root = trie._deserialize(data)
        return trie

    def _serialize(self, node):
        data = {
            'end_of_word': node.end_of_word,
            'count': node.count,
            'children': {}
        }
        for char, child_node in node.children.items():
            data['children'][char] = self._serialize(child_node)
        return data

    def _deserialize(self, data):
        node = TrieNode()
        node.end_of_word = data['end_of_word']
        node.count = data['count']
        for char, child_data in data['children'].items():
            node.children[char] = self._deserialize(child_data)
        return node
        
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
        if node.end_of_word and word in prefix:
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


def setup():
    lines = read_all()
    
    trie = ContainTrie()
    
    for word in lines:
        trie.insert(word)
    return trie

if __name__ == '__main__':
    pre_pt_filename = 'contain.json'
    
    if os.path.exists(pre_pt_filename):
        contain_model = ContainTrie.load(pre_pt_filename)
    else:
        contain_model = setup()
        contain_model.save(pre_pt_filename)
        
    
    # for loop input
    while True:
        prefix = input("Input:")
        
        # results = trie.search(prefix)
        results = []
        # for word in prefix:
        results.extend(contain_model.get_strings_with_word(word=prefix))
        
        if len(results) == 0:
            print('Not found.\n')
        else:
            # print("Results:{}".format(results[0][0]))
            print("Results:\n- {}".format('\n- '.join([x[0] for x in results[:5]])))
            print()
            