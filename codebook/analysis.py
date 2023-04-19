import argparse
from collections import defaultdict
from utils import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/test_1500.txt', type=str, required=False, help='Input file path in txt format.')
    # parser.add_argument('--output', default='../data/test_1500.json', type=str, required=False,
                        # help='Output file in json format.')
    
    args = parser.parse_args()
    
    in_file = args.input
    # out_file = args.output
    
    lines = open(in_file, 'r').readlines()
    
    print(f'Read {len(lines)} lines.')
    
    sents = defaultdict(int)
    
    for line in lines:
        line = preprocess(line)
        if len(line) == 0:
                continue
        
        for sent in line.split('。'):
            if len(sent):
                sents[sent] += 1
    
    sorted_sents = sorted(sents.items(), key=lambda x:x[1], reverse=True)
    for st, v in sorted_sents:
        if v > 10:
            print(st, v)
    