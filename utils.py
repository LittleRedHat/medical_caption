import unicodedata
import re

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'[\s]+[^a-zA-Z]+[\s]+',r' ',s)
    s = re.sub(r'dr\.',r'dr',s)
    s = re.sub(r'^[0-9]\.',r"",s)
    s = re.sub(r"[\s]+[0-9]\.",r"",s)
    s = re.sub(r"([.!?][\s]+)", r" \1 ", s)   #将标点符号用空格分开
    s = re.sub(r"([.!?])$", r" \1 ", s)

    s = re.sub(r"[^a-zA-Z0-9\.!\?'\-/]+", r" ", s)  #除字母标点符号数字的其他连续字符替换成一个空格
    return s

def normalize_word(s):
    pass

if __name__ == '__main__':
    s = "1. Left lower lobe nodule which is worrisome. If there are no prior films available for comparison XXXX scan for further evaluation"
    normalize_string(s)
    
