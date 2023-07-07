import json
from code.TRIE import Trie, TrieNode

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
