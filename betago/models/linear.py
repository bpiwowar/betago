from pathlib import Path
import logging

import torch
from torch.autograd import Variable

from betago.model import BaseModel, TorchModel
from betago.commands import argument


# Arguments can be added using the "argument" annotation
# whose parameters follow the argpase.ArgumentParser.add_argument
@argument("--hidden", type=int, default=0)
class Model(TorchModel):
    """
    A simple model with one or two layers
    """
    def construct(self):
        """Called when the model has been configured to construct the network"""
        self.input_size = self.numplanes * self.boardsize**2
        if self.hidden:
            self.modules = torch.nn.ModuleList([
                torch.nn.Linear(self.input_size, self.hidden), 
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden, self.boardsize**2)
            ])
        else:
            self.layers = torch.nn.Linear(self.input_size, self.boardsize**2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        logging.info("Model initialized: %s", self)


    def _predict(self, boards, volatile=True):
        _boards = Variable(torch.Tensor(boards), volatile=volatile)
        y = self.modules(_boards.view(-1, self.input_size))
        return y

    def _cost(self, boards, labels, volatile=True):
        _labels = Variable(torch.LongTensor(labels), volatile=volatile)
        y = self._predict(boards, volatile=volatile)
        return torch.nn.functional.cross_entropy(y, _labels)

    def predict(self, board):
        y = self._predict(board)
        return torch.nn.functional.softmax(y).data.numpy()

    def train(self, boards, labels):
        """Train the model on the batch and returns the current cost"""
        cost = self._cost(boards, labels, volatile=False)
        cost.backward()
        self.optimizer.step()
        
        self.epoch += 1
        return cost.data.numpy()

    def cost(self, boards, labels):
        return self._cost(boards, labels, volatile=True).data.numpy()
        

