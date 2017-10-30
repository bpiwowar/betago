from pathlib import Path
import logging

import torch
from torch.autograd import Variable

from gammago.model import BaseModel, TorchModel
from gammago.commands import argument


# Arguments can be added using the "argument" annotation
# whose parameters follow the argpase.ArgumentParser.add_argument
@argument("--hidden", type=int, default=0)
class Model(TorchModel):
    """
    A simple model with one or two layers
    
    Torch models must implement four methods:
    - construct: construct the module
    - train: train with a batch and output the cost
    - cost: compute the cost of a batch
    - predict: 
    """
    def construct(self):
        """Called when the model has been configured to construct the network"""
        self.input_size = self.numplanes * self.boardsize**2
        
        if self.hidden:
            self.layers = torch.nn.ModuleList([
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
        y = _boards.view(-1, self.input_size)
        for layer in self.layers:
            y = layer(y)
        return y

    def _cost(self, boards, labels, volatile=True):
        _labels = Variable(torch.LongTensor(labels), volatile=volatile)
        y = self._predict(boards, volatile=volatile)
        return torch.nn.functional.cross_entropy(y, _labels)

    def predict(self, boards):
        """Predict a move
        
        :param boards: a numpy tensor of dimension (batch size, num planes, board size, board size)
        :returns: A probability distribution over moves
        """
        y = self._predict(boards)
        return torch.nn.functional.softmax(y).data.numpy()

    def train(self, boards, labels):
        """Train the model on the batch and returns the current cost
    
        :param boards: a numpy real tensor of dimension (batch size, num planes, board size, board size)
        :param labels: a numpy integer tensor of dimension (batch size, output size)

        """
        cost = self._cost(boards, labels, volatile=False)
        cost.backward()
        self.optimizer.step()
        
        self.epoch += 1
        return cost.data.numpy()

    def cost(self, boards, labels):
        """Return the batch cost
    
        :param boards: a numpy real tensor of dimension (batch size, num planes, board size, board size)
        :param labels: a numpy integer tensor of dimension (batch size, output size)

        """
        return self._cost(boards, labels, volatile=True).data.numpy()
        

