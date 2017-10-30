import numpy as np

from gammago.model import BaseModel
from gammago.processor import ThreePlaneProcessor

class Model(BaseModel):
    '''
    Play random moves, like a good 30k bot.
    '''
    def __init__(self):
        super().__init__()
        self.processor = ThreePlaneProcessor()

    def predict(self, board):
        size = board.shape[-1] ** 2
        p = np.random.randn(1, size)
        p = np.abs(p) + 1e-8
        p /= p.sum()
        return p
