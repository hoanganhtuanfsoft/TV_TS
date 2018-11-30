import gensim
import os
from pyvi import ViTokenizer
from gensim.models.callbacks import CallbackAny2Vec
import string
import time
import sys

import sys
sys.path.append('../../../')
import TV_Utility

class MyDocuments:
    def __init__(self, dirname, ext_filter=['.txt'], encoding='utf-8', language='vi', is_remove_stopword=True):
        self.__dirname = dirname
        self.__ext_filter   = ext_filter
        self.__encoding     = encoding
        self.__language     = language
        self.__is_remove_stopword   = is_remove_stopword


    def __iter__(self):
        """ yeild each doc and tags in dataset during train time
            Returns:
                words (list of str): tokens in a document
                [tag] (list of int): list of tag assign to document
        """
        path_tags = self.__get_file_paths_and_tags(self.__dirname)
        for (fpath, tag) in path_tags.items():
            if not os.path.splitext(fpath)[-1] in self.__ext_filter:
                continue
            try:
                words = []
                for line in open(fpath, encoding=self.__encoding):
                    if len(line) == 0:
                        continue
                    line   = TV_Utility.pre_process_sentence([line], language=self.__language, is_remove_stopword=self.__is_remove_stopword)[0]
                    tokens = ViTokenizer.tokenize(line).split()
                    words  = words + tokens
                yield gensim.models.doc2vec.TaggedDocument(words, [tag])
            except Exception as e:
                print(e, fpath)

    def __get_file_paths_and_tags(self, root):
        """ Get all files path in folder and assign each path with a unique number
            Args:
                root (str): location of root folder
            Returns:
                path_tags (dict of path:id): list of all files path in {root} folder
                and its ID
        """
        def __get_file(__root, __path_tags, __tag):
            for fname in os.listdir(__root):
                full_path = os.path.join(__root, fname)
                if os.path.isdir(full_path):
                    __path_tags, __tag = __get_file(full_path, __path_tags, __tag)
                else:
                    __path_tags[full_path] = __tag
                    __tag += 1
            return __path_tags, __tag

        tag = 0
        path_tags = {}
        path_tags, tag = __get_file(root, path_tags, tag)
        return path_tags

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        self.__start = time.time()
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        t = time.time() - self.__start
        print("Epoch #{} end".format(self.epoch))
        print("Time = ", t)
        self.epoch += 1

if __name__ == '__main__':
    MySentences('../../../')