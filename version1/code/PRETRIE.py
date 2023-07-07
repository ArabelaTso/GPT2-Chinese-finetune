import json
from code.TRIE import Trie, TrieNode

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
    
    