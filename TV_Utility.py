import string
import nltk
import re
from pyvi import ViTokenizer

import TV_Config

def split_words(sent, language = 'vi'):
    # change format to split words if current language is VN
    if language == 'vi':
        sent = ViTokenizer.tokenize(sent)

    return nltk.tokenize.word_tokenize(sent)

# remove html tag
def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

# Read list of stop words from the pre-define file
def read_stopwords(file_path):
    with open(file_path, 'r') as f:
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
def pre_process_sentence(sents, language = 'vi'):
    for i in range(len(sents)):
        # remove html tag
        sents[i] = clean_html(sents[i])

        # change to lower case
        sents[i] = sents[i].lower()

        # Get words
        words = split_words(sents[i])

        #remove stop words
        words = remove_stopwords(words)

        # stem words
        if language != 'vi':
            words = stem_words(words)

        sents[i] = ' '.join(words)

    return sents

if __name__ == '__main__':

    print(pre_process_sentence(['works working '],language = 'en'))

