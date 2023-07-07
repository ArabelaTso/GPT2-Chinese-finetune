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