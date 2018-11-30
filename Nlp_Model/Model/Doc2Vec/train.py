import config
import util

import os
import sys
import gensim 

## checking output path first
if not os.path.isdir(os.path.dirname(config.OUTPUT_VECTOR_PATH)):
    print('The folder is not exist: ', config.OUTPUT_VECTOR_PATH)
    sys.exit()
if not os.path.isdir(os.path.dirname(config.OUTPUT_MODEL_PATH)):
    print('The folder is not exist: ', config.OUTPUT_MODEL_PATH)
    sys.exit()

sentences = util.MyDocuments(dirname=config.DATASET, ext_filter=config.FILE_EXT, encoding=config.FILE_ENCODING, language=config.LANGUAGE, is_remove_stopword=config.IS_REMOVE_STOP_WORDS)
epoch_logger = util.EpochLogger()

print("Training")
## checking if existing training model
if os.path.exists(config.OUTPUT_MODEL_PATH):
    model   = gensim.models.doc2vec.Doc2Vec.load(config.OUTPUT_MODEL_PATH)
    model.train(sentences, total_words=len(model.wv.vocab), epochs=config.EPOCH, callbacks=[epoch_logger])
else:
    model   = gensim.models.doc2vec.Doc2Vec(sentences, dm=config.USE_PV_DM, epochs=config.EPOCH, size=config.VECTOR_SIZE, window=config.WINDOW, callbacks=[epoch_logger])

print("Saving model")
model.save(config.OUTPUT_MODEL_PATH)
model.wv.save(config.OUTPUT_VECTOR_PATH)
print("Done")

## test
from pyvi import ViTokenizer
text = 'Nó được quản lý bởi Bộ Bưu điện & Viễn thông Bangladesh'
text = ViTokenizer.tokenize(text).split()
print(model.infer_vector(text))

