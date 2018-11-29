from Nlp_Model.Model.Sent2Vec_Networks import *
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
INPUT_DATA = os.path.join(DIR_PATH,'Nlp_Model/Data/data_news.txt')
SAVED_MODEL = os.path.join(DIR_PATH,'Saved_model/sent2vec_model')
obj = Sent2Vec_Networks(input_data=INPUT_DATA,saved_model_loc=SAVED_MODEL)
obj.train()