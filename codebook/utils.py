import re

def preprocess(line):
    line = line.rstrip()
    line = line.lstrip()
    line = line.replace("\"", "")
    line = line.replace('嘱：', '')
    line = line.replace("既往史:", "")
    line = line.replace("主诉：", "")
    line = re.sub(r'\b-', '', line)
    line = line.replace('；', '。')
    line = re.sub(r'\b\d[，\.）、]\s?', '', line)
    line = re.sub(r"Frankl \d-\d\s?，?", '', line)
    line = re.sub(r"Frankl \d\s?，?", '', line)
    line = line.replace('Frankl', '')
    line = re.sub(r'\b\d，', '', line)
    line = re.sub(r'OHI[、，]', '', line)
    line = line.replace('\t', '')
    line = line.replace('\\', ' ')
    line = line.replace(' ', '')
    line = line.replace('*', '')
    line = line.replace('“', '')
    line = line.replace('”', '')
    
    line = re.sub(r'（.*?）', '', line)
    
    return line