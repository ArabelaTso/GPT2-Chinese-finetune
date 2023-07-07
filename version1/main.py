import os
import sys
import json
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_path))
sys.path.append('./')

from collections import defaultdict
from code.PRETRIE import PrefixTrie
from code.CONTRIE import ContainTrie
from code.utils import *



if __name__ == '__main__':
    prefix_model, contain_model = setup_models()
    
    # for loop input
    while True:
        input_str = input("\n输入 (输入 'q' 结束循环): ")
        
        if input_str == 'q':
            print('感谢使用.')
            break
        
        # Completion
        complete_results = prefix_model.search(input_str)
        if len(complete_results) == 0:
            complete_results = '无相关语句可补全.'
        else:
            complete_results = complete_results[0][0]
        
        # Contain
        contain_results = []
        contain_results.extend(contain_model.get_strings_with_word(word=input_str))
        
        if len(contain_results) == 0:
            contain_results = [('无相关联想.', 0)]
        else:
            contain_results = contain_results[:min(len(contain_results), 5)]
        
        print("补全：{}\n联想：\n- {}\n\n".format(complete_results, '\n- '.join([x[0] for x in contain_results])))
        

