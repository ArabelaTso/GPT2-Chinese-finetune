
if __name__ == '__main__':
    in_file = './test_1500.txt'
    out_file = './test_1500_try.json'
    
    lines = open(in_file, 'r').readlines()
    
    print(f'Read {len(lines)} lines.')
    lines = lines[:2]
    
    with open(out_file, 'w') as fw:
        fw.write("[\"")
        for line in lines:
            line = line.replace("\"", "")
            fw.write(line + '\n')
        fw.write("\"]")
        # fw.write("[\"{}\"]".format(''.join([line.replace("\"", "", line) for line in lines])))
        # fw.write('[\"' + '\n'.join([line.replace("\"", "", line) for line in lines] + '\"]'))
    