import re

def preprocess(line):
    line = line.rstrip()
    line = line.lstrip()
    line = line.replace("\"", "")
    line = line.replace('嘱：', '')
    if "局麻麻木感通常3个小时" in line:
        print(str(line))
    line = re.sub(r'^\d[，.]\s?', '', line)
    line = re.sub(r"Frankl \d-\d\s?，?", '', line)
    line = re.sub(r"Frankl \d\s?，?", '', line)
    line = line.replace('\t', '')
    line = line.replace('\\', ' ')
    line = line.replace(' ', '')

    return line