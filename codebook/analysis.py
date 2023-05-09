import argparse
from collections import defaultdict
from utils import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/train_all.txt', type=str, required=False, help='Input file path in txt format.')
    # parser.add_argument('--output', default='../data/test_1500.json', type=str, required=False,
                        # help='Output file in json format.')
    
    args = parser.parse_args()
    
    in_file = args.input
    # out_file = args.output
    
    thr_cnt = 1
    thr_len = 1
    
    lines = open(in_file, 'r').readlines()
    
    print(f'Read {len(lines)} lines.')
    
    sent_count_dict = defaultdict(int)
    sent_all = []
    
    for line in lines:
        line = preprocess(line)
        if len(line) == 0:
                continue
        
        for sent in line.split('。'):
            if len(sent):
                sent_count_dict[sent] += 1
                sent_all.append(sent)
                for phrase in sent.split('，'):
                    sent_all.append(phrase)
                    
    
    # Output statistics, a sentence followed by its counts
    # Sort dictionary by descending counts
    # sorted_sent_count_dict = sorted(sent_count_dict.items(), key=lambda x:x[1], reverse=True)
    # for st, v in sorted_sent_count_dict:
    #     if len(st) <= 1:
    #         continue
    #     if v > 5:
    #         print(st)
    #         # print(st, v)
    
    # Output all sentence, without removing the duplication
    for sent in sent_all:
        if len(sent) > 1:
            print(sent)