
if __name__ == '__main__':
    in_file = './test_1500.txt'
    out_file = './test_1500_try.json'
    
    lines = open(in_file, 'r').readlines()
    
    print(f'Read {len(lines)} lines.')
    
    with open(out_file, 'w') as fw:
        fw.write("[\"")
        for line in lines:
            line = line.rstrip()
            line = line.replace("\"", "")
            line = line + "\\n"
            fw.write(line)
        fw.write("\"]")
    