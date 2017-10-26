from pathlib import Path
import logging

import torch
from torch.autograd import Variable

from betago.model import BaseModel, TorchModel


class Model(TorchModel):
    def __init__(self, arguments, parameterspath: Path):
        if arguments:
            raise Exception("No extra arguments needed")
        super().__init__(parameterspath)

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        
        self.input_size = self.numplanes * self.boardsize**2
        self.linear = torch.nn.Linear(self.input_size, self.boardsize**2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
    

    def _predict(self, boards, volatile=True):
        _boards = Variable(torch.Tensor(boards), volatile=volatile)
        y = self.linear(_boards.view(-1, self.input_size))
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
        

