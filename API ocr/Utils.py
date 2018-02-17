
LETTERS = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def labels_to_text(labels):
    return ''.join(list(map(lambda x: LETTERS[int(x)], labels)))

def text_to_labels(text, max_len):
    labels =  list(map(lambda x: LETTERS.index(x), text))
    for i in range(max_len - len(labels)):
        labels.append(1)
    return labels

def is_valid_str(s):
    for ch in s:
        if not ch in LETTERS:
            return False
    return True