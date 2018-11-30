"""
Example to load pretrained model
Doc at https://radimrehurek.com/gensim/models/word2vec.html
"""

import gensim

if __name__ == '__main__':
    model = gensim.models.KeyedVectors.load('../../Saved_model/w2v_include_stop_word_11302018')
    print(model.wv['hoa'])