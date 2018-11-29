import torch
from torch import nn
from torch.autograd import Variable

from datetime import datetime, timedelta

from .Sent2Vec.skip_thoughts.data_loader import DataLoader
from .Sent2Vec.skip_thoughts.model import UniSkip
from .Sent2Vec.skip_thoughts.config import *

class Skip_Thoughts:
    def __init__(self, batch_size=64, epochs=1000, input_data='../Data/data.txt',
                 saved_model_loc='../../Saved_model/sent2vec_model'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.saved_model_loc = saved_model_loc
        self.d = DataLoader(input_data)
        self.model = UniSkip()
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)

        self.loss_trail = []
        self.last_best_loss = None
        self.current_time = datetime.utcnow()

    def debug(self, i, loss, prev, nex, prev_pred, next_pred):

        this_loss = loss.item()
        self.loss_trail.append(this_loss)
        self.loss_trail = self.loss_trail[-20:]
        new_current_time = datetime.utcnow()
        time_elapsed = str(new_current_time - self.current_time)
        self.current_time = new_current_time
        print("Iteration {}: time = {} last_best_loss = {}, this_loss = {}".format(i, time_elapsed, self.last_best_loss,
                                                                                   this_loss))

        print("prev = {}\nnext = {}\npred_prev = {}\npred_next = {}".format(
            self.d.convert_indices_to_sentences(prev),
            self.d.convert_indices_to_sentences(nex),
            self.d.convert_indices_to_sentences(prev_pred),
            self.d.convert_indices_to_sentences(next_pred),
        ))

        try:
            trail_loss = sum(self.loss_trail) / len(self.loss_trail)
            if self.last_best_loss is None or self.last_best_loss > trail_loss:
                print("Loss improved from {} to {}".format(self.last_best_loss, trail_loss))

                print("saving model at {}".format(self.saved_model_loc))
                torch.save(self.model.state_dict(), self.saved_model_loc)

                self.last_best_loss = trail_loss
        except Exception as e:
            print("Couldn't save model because {}".format(e))

    def train(self):

        print("Starting to train Skip_thoughts...")
        for i in range(self.epochs):
            sentences, lengths = self.d.fetch_batch(self.batch_size)

            loss, prev, nex, prev_pred, next_pred = self.model(sentences, lengths)

            if i % 10 == 0:
                self.debug(i, loss, prev, nex, prev_pred, next_pred)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class Sent2Vec_Networks:
    def __init__(self, batch_size=64, epochs=1000, input_data='../Data/data.txt',
                 saved_model_loc='../../Saved_model/sent2vec_model', algorithm='skip_thoughts'):
        self.model = None
        if algorithm == 'skip_thoughts':
            self.model = Skip_Thoughts(batch_size, epochs, input_data, saved_model_loc)

    def train(self):
        if self.model is not None:
            self.model.train()
