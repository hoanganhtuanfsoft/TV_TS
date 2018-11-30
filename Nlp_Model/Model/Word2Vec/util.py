import gensim
import os
from pyvi import ViTokenizer
from gensim.models.callbacks import CallbackAny2Vec
import string
import time
import sys

sys.path.append('../../../')
import TV_Utility

class MySentences:
    def __init__(self, dirname, ext_filter=['.txt'], encoding='utf-8', language='vi', is_remove_stopword=True):
        self.__dirname      = dirname
        self.__ext_filter   = ext_filter
        self.__encoding     = encoding
        self.__language     = language
        self.__is_remove_stopword   = is_remove_stopword

    def __iter__(self):
        """ yeild each line of text in dataset during train time
            Returns:
                tokens (list of str): tokens in a line of text
        """
        paths = self.__get_file_paths(self.__dirname)
        for fpath in paths:
            if not os.path.splitext(fpath)[-1] in self.__ext_filter:
                continue
            try:
                for line in open(fpath, encoding=self.__encoding):
                    if len(line) == 0:
                        continue
                    line   = TV_Utility.pre_process_sentence([line], language=self.__language, is_remove_stopword=self.__is_remove_stopword)[0]
                    tokens = ViTokenizer.tokenize(line).split()
                    if len(tokens) == 0:
                        continue
                    yield tokens
            except Exception as e:
                print(e, fpath)

    def __get_file_paths(self, root):
        """ Get all files path in folder
            Args:
                root (str): location of root folder
            Returns:
                paths (list of str): list of all files path in {root} folder
        """
        def __get_file(__root, __paths):
            for fname in os.listdir(__root):
                full_path = os.path.join(__root, fname)
                if os.path.isdir(full_path):
                    __paths = __get_file(full_path, __paths)
                else:
                    __paths.append(full_path)
            return __paths

        paths   = []
        paths   = __get_file(root, paths)
        return paths

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
