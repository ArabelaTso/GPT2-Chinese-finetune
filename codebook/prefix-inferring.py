import sys

from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_word = False
        self.count = 0

class PrefixTrie:
    def __init__(self):
        self.root = TrieNode()

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
    
def read_all():
    lines = open('./data/clean/train_all.txt', 'r').readlines()
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

if __name__ == '__main__':
    trie = PrefixTrie()
    lines = read_all()
    # print(lines[:2])
    
    words = lines
    # words = ["apple", "banana", "orange", "pear", "pineapple", "apple", "banana", "orange"]
    for word in words:
        trie.insert(word)

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
        

