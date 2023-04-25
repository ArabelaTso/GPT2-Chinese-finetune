import re
import sys
import time
import curses
from difflib import get_close_matches

def read_strings(file='./data/stats/stats_all.txt'):
    strings = open(file, 'r').readlines()
    strings = strings[1:]
    strings = list(map(lambda x: x.replace('\n', ''), strings))
    return strings

def find_strings_with_prefix(prefix, strings):
    candidates = []
    for st in strings:
        if st.startswith(prefix):
            # yield postprocess(st)
            candidates.append(postprocess(st))
    return candidates

def find_bestmatch(st, strings):
    return get_close_matches('定期', ['定期检查充填物情况，脱落不适随诊', '定期检查', '定期检查充填物', '定期检查充填物情况'])
    # return get_close_matches(st, strings)

def postprocess(st):
    return re.sub(r'【\d】', '【 】', st)


def report_progress(stdscr, st, progress):
    """progress: 0-10"""
    stdscr.addstr(progress, 0, "{0}".format(st))
    # stdscr.addstr(1, 0, "Total progress: [{1:10}] {0}%".format(progress * 10, "#" * progress))
    stdscr.refresh()

def main():
    strings = read_strings()
    
    # stdscr = curses.initscr()
    # curses.noecho()
    # curses.cbreak()
    
    # try:
    #     prefix = input("Input:")
    #     candidates = find_strings_with_prefix(prefix, strings)
    #     for i, st in enumerate(candidates):
    #         report_progress(stdscr, st, i)
    #         time.sleep(0.5)
    # finally:
    #     curses.echo()
    #     curses.nocbreak()
    #     curses.endwin()
    
    while True:
        prefix = input("Input:")
        candidates = find_strings_with_prefix(prefix, strings)
        # candidates = candidates[:min(1, len(candidates))]
        # candidates = find_bestmatch(prefix, strings)
        
        print('\n'.join(candidates))
        print()
        # for st in candidates:
        #     print('\r{}\n'.format(st), end='\n')
        #     time.sleep(0.5)
        

if __name__ == '__main__':
    main()