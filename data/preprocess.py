
if __name__ == '__main__':
    in_file = './data/test_1500.txt'
    out_file = './data/test_1500_try.json'
    
    lines = open(in_file, 'r').readlines()
    
    print(f'Read {len(lines)} lines.')
    with open(out_file, 'w') as fw:
        fw.write("[\"{}\"]".format('\n'.join(lines)))
        # fw.write('[\"' + '\n'.join([line.replace("\"", "", line) for line in lines] + '\"]'))
    