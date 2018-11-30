import string
import nltk
import re
from pyvi import ViTokenizer
import string

import TV_Config

def split_words(sent, language = 'vi'):
    # change format to split words if current language is VN
    if language == 'vi':
        sent = ViTokenizer.tokenize(sent)

    return nltk.tokenize.word_tokenize(sent)

# Remove hyperlink
def remove_hyperlinks(sent):
    sent = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', sent, flags=re.MULTILINE)
    return sent

# remove html tag
def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

# Read list of stop words from the pre-define file
def read_stopwords(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    return stopwords

# Based on language, get stop words and remove it.
def remove_stopwords(words, language = 'vi'):
    if language == 'vi':
        stopwords = set(read_stopwords(TV_Config.STOP_WORDS))
    else:
        stopwords = set(nltk.corpus.stopwords.words('english'))

   # words = nltk.tokenize.word_tokenize(sent)
    ''' If remove special character, use the following code 
    split_words = [x.strip(TV_Config.SPECIAL_CHARACTER).lower() for x in split_words]
    return [word for word in split_words if word.encode('utf-8') not in stopwords]'''

    return [word for word in words if word not in stopwords]

def stem_words(words):
    porter = nltk.stem.porter.PorterStemmer()
    return [porter.stem(word) for word in words]

# Input: sents: list of sentences
# Output: list of preprocessed sentences
def pre_process_sentence(sents, language = 'vi', is_remove_stopword=True, is_remove_hyperlink=True, is_remove_punctuation=True, is_remove_digit=True):
    for i in range(len(sents)):
        # remove hyperlink
        if is_remove_hyperlink:
            sents[i]    = remove_hyperlinks(sents[i])

        # remove html tag
        sents[i] = clean_html(sents[i])
        # remove punctuation
        if is_remove_punctuation:
            sents[i] = ''.join([ch for ch in sents[i] if ch not in string.punctuation and ch != '–'])

        # change to lower case
        sents[i] = sents[i].lower()

        # Get words
        words = split_words(sents[i])

        #remove stop words
        if is_remove_stopword:
            words = remove_stopwords(words)
        # stem words
        if language != 'vi':
            words = stem_words(words)
        sents[i] = ' '.join(words)

        # remove digits
        if is_remove_digit:
            sents[i] = ' '.join([token for token in sents[i].split() if not token.isdigit()])

    return sents

if __name__ == '__main__':
    print(pre_process_sentence(['works iphone5s working https://www.dailymail.co.uk/home/index.html 65  == ?something 65'],language = 'vi'))

