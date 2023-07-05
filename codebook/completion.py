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

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        trie = Trie()
        trie.root = trie._deserialize(data)
        return trie

    def save(self, file_path):
        data = self._serialize(self.root)
        with open(file_path, 'w') as f:
            json.dump(data, f)
            
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
    
    
class ContainTrie(Trie):
    def __init__(self):
        super().__init__()

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

def sort_list_by_second_element(lst):
    """
    Sorts a list according to the second element in every element in the list.
    
    Args:
    lst: list of tuples, where each tuple contains at least two elements
    
    Returns:
    A sorted list according to the second element in every element in the list.
    """
    return sorted(lst, key=lambda x: x[1], reverse=True)

def setup():
    trie = PrefixTrie()
    lines = read_all()
    
    for word in lines:
        trie.insert(word)
    return trie

if __name__ == '__main__':
    trie = setup()

    # for loop input
    while True:
        prefix = input("Input:")
        results = trie.search(prefix)
        if len(results) == 0:
            print('Not found.\n')
        else:
            print("Results:{}".format(results[0][0]))
            print()
            
            # for res in results[0]:
                # print(res)
    
    # for once
    # prefix = "光"
    # results = trie.search(prefix)
    # sorted_results = sort_list_by_second_element(results)
    # print(sorted_results[0][0])
    # for res in  sorted_results:
        # print(res)
        

