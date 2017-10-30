# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
import os
import random
from .index_processor import KGSIndex



class SingleSampler:
    def __init__(self, string):
        a = [int(x) if x else None for x in string.split(":")]
        self.game_ratio = a.pop(0) / 100
        self.game_max = a.pop(0) if a else None
        self.board_ratio = a.pop(0) / 100 if a else 100
        self.board_max = a.pop(0) if a else None
        assert self.game_ratio <= 100 and self.game_ratio >= 0
        assert self.board_ratio <= 100 and self.game_ratio >= 0
        
        self.board_count = 0
        self.game_count = 0

    def sample_game(self):
        if not self.game_max: 
            return True
        if self.game_count < self.game_max:
            self.game_count += 1
            return True
        return False
    
    def sample_board(self):
        if not self.board_max: 
            return True
        if self.board_count < self.board_max:
            self.board_count += 1
            return True
        return False
            
    
    def __repr__(self):
        return "%.d%% games (max %s), %d%% boards (max %s)" % \
            (self.game_ratio * 100, self.game_max, self.board_ratio * 100, self.board_max)

    def __str__(self):
        s = "%.d" % (self.game_ratio * 100)
        if self.game_max: s += "-" + str(self.game_max)
        s += "_%d" % (self.board_ratio * 100)
        if self.board_max: s += "-" + str(self.board_max)
        return s

class Sampler(object):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    IGNORE = 3

    '''
    Sample training and test data from zipped sgf files such that test data is kept stable.
    '''
    def __init__(self, train: SingleSampler, val: SingleSampler, test: SingleSampler, *, 
        seed=0):
        self.samplers = [train, val, test]
        self.random = random.Random(seed)

    def game(self):
        '''Returns whether the game should be sampled for train, val, test or ignored'''
        x = self.random.uniform(0,1)

        for mode in [Sampler.TRAIN, Sampler.VALIDATION, Sampler.TEST]:
            x -= self.samplers[mode].game_ratio
            if x < 0: 
                if self.samplers[mode].sample_game():
                    return mode
                return Sampler.IGNORE            

        return Sampler.IGNORE

    def sample_board(self, mode):
        if self.random.uniform(0,1) < self.samplers[mode].board_ratio:
            return self.samplers[mode].sample_board()
        return False



