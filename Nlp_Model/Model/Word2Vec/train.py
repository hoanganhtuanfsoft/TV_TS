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

sentences = util.MySentences(dirname=config.DATASET, ext_filter=config.FILE_EXT, encoding=config.FILE_ENCODING, language=config.LANGUAGE, is_remove_stopword=config.IS_REMOVE_STOP_WORDS)
epoch_logger = util.EpochLogger()

print("Training")
## checking if existing training model
if os.path.exists(config.OUTPUT_MODEL_PATH):
    model   = gensim.models.Word2Vec.load(config.OUTPUT_MODEL_PATH)
    model.train(sentences, total_words=len(model.wv.vocab), epochs=config.EPOCH, callbacks=[epoch_logger])
else:
    model   = gensim.models.Word2Vec(sentences, sg=config.SKIP_GRAM, size=config.VECTOR_SIZE, iter=config.EPOCH, callbacks=[epoch_logger])

print("Saving model")
model.save(config.OUTPUT_MODEL_PATH)
model.wv.save(config.OUTPUT_VECTOR_PATH)
print("Done")