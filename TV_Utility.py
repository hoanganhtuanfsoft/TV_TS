import string
import nltk
import re
from pyvi import ViTokenizer

import TV_Config

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
def remove_stopwords(sent, language = 'vi'):
    if language == 'vi':
        stopwords = set(read_stopwords(TV_Config.STOP_WORDS))
        sent = ViTokenizer.tokenize(sent)
        split_words = sent.split()
    else:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        split_words = nltk.tokenize.word_tokenize(sent)


    try:
        ''' If remove special character, use the following code 
        split_words = [x.strip(TV_Config.SPECIAL_CHARACTER).lower() for x in split_words]
        return [word for word in split_words if word.encode('utf-8') not in stopwords]'''

        return [word for word in split_words if word not in stopwords]
    except TypeError:
        return []


def stem_words(sent):
    split_words = nltk.tokenize.word_tokenize(sent)
    porter = nltk.stem.porter.PorterStemmer()
    return [porter.stem(word) for word in split_words]

# Input: sents: list of sentences
# Output: list of preprocessed sentences
def pre_process_sentence(sents, language = 'vi'):
    for i in range(len(sents)):
        # remove html tag
        sents[i] = clean_html(sents[i])

        # remove multiple spaces
        sents[i] = ' '.join(sents[i].split())

        # change to lower case
        sents[i] = sents[i].lower()

        # remove stop words
        sents[i] = ' '.join(remove_stopwords(sents[i], language))

        # stem words
        if language != 'vi':
            sents[i] = ' '.join(stem_words(sents[i]))

    return sents

if __name__ == '__main__':

    print(pre_process_sentence(['works working '],language = 'en'))

