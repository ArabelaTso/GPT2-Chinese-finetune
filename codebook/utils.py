import re

def preprocess(line):
    line = line.rstrip()
    line = re.sub(r"Frankl \d-\d\s?，?", '', line)
    line = re.sub(r"Frankl \d\s?，?", '', line)
    line = line.replace('\t', '')
    line = line.replace('\\', ' ')
    line = line.replace("\"", "")
    line = line.replace(' ', '')

    return line