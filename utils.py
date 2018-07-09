import unicodedata
import re

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1 ", s)   #将标点符号用空格分开
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  #除字母标点符号的其他连续字符替换成一个空格
    return s
