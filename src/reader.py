import re

class Reader():

    def __init__(self, path):
        self.path = path

    def get_corpus(self):
        print("\n L O A D I N G - C O R P U S\n")
        f = open(self.path, 'r', encoding='latin-1')
        text = f.read()
        f.close()

# All the text to lowercase
        text = text.lower()
# Delete strange characters
        text = re.sub(r'[^\x00-\x7f]', r'', text)
# Delete extra spaces
        text = re.sub(r'\s{2,}', r'', text)
# Delete new lines
        text = re.sub(r'\n+', r'', text)
        return text
