from __future__ import absolute_import
import numpy as np
from .dataloader.base_processor import GoDataProcessor
from six.moves import range
import struct

class SevenPlaneProcessor(GoDataProcessor):
    '''
    Implementation of a Go data processor, using seven planes of 19x19 values to represent the position of
    a go board, as explained below.

    This closely reflects the representation suggested in Clark, Storkey:
    http://arxiv.org/abs/1412.3409
    '''

    def __init__(self, data_directory='data', num_planes=7, use_generator=False):
        super(SevenPlaneProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  use_generator=use_generator)

    def feature_and_label(self, color, move, go_board):
        '''
        Parameters
        ----------
        color: color of the next person to move
        move: move they decided to make
        go_board: represents the state of the board before they moved

        Planes we write:
        0: our stones with 1 liberty
        1: our stones with 2 liberty
        2: our stones with 3 or more liberties
        3: their stones with 1 liberty
        4: their stones with 2 liberty
        5: their stones with 3 or more liberties
        6: simple ko
        '''
        row, col = move
        enemy_color = go_board.other_color(color)
        label = row * 19 + col
        move_array = np.zeros((self.num_planes, go_board.board_size, go_board.board_size))
        for row in range(0, go_board.board_size):
            for col in range(0, go_board.board_size):
                pos = (row, col)
                if go_board.board.get(pos) == color:
                    if go_board.go_strings[pos].liberties.size() == 1:
                        move_array[0, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() == 2:
                        move_array[1, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() >= 3:
                        move_array[2, row, col] = 1
                if go_board.board.get(pos) == enemy_color:
                    if go_board.go_strings[pos].liberties.size() == 1:
                        move_array[3, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() == 2:
                        move_array[4, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() >= 3:
                        move_array[5, row, col] = 1
                if go_board.is_simple_ko(color, pos):
                    move_array[6, row, col] = 1
        return move_array, label

    def store_results(self, data_file, color, move, go_board):
        '''
        Parameters
        ----------
        color: color of the next person to move
        move: move they decided to make
        go_board: represents the state of the board before they moved

        Planes we write:
        0: our stones with 1 liberty
        1: our stones with 2 liberty
        2: our stones with 3 or more liberties
        3: their stones with 1 liberty
        4: their stones with 2 liberty
        5: their stones with 3 or more liberties
        6: simple ko
        '''
        row, col = move
        enemy_color = go_board.other_color(color)
        data_file.write(BinaryRepresentation.POSITION_HEADER)
        label = row * go_board.board_size + col
        data_file.write(BinaryRepresentation.LABELSTRUCT.pack(label))

        # 1 byte for each cell of the board
        for row in range(0, go_board.board_size):
            for col in range(0, go_board.board_size):
                value = 0               
                pos = (row, col)

                if go_board.board.get(pos) == color:
                    if go_board.go_strings[pos].liberties.size() == 1:
                        value += 1 << 7
                    elif go_board.go_strings[pos].liberties.size() == 2:
                        value += 1 << 6
                    elif go_board.go_strings[pos].liberties.size() >= 3:
                        value += 1 << 5
                    
                elif go_board.board.get(pos) == enemy_color:
                    if go_board.go_strings[pos].liberties.size() == 1:
                        value += 1 << 4
                    elif go_board.go_strings[pos].liberties.size() == 2:
                        value += 1 << 3
                    elif go_board.go_strings[pos].liberties.size() >= 3:
                        value += 1 << 2

                if go_board.is_simple_ko(color, pos):
                    value += 1 << 1

                # write in file
                data_file.write(bytes([value]))

class ThreePlaneProcessor(GoDataProcessor):
    '''
    Simpler version of the above processor using just three planes. This data processor uses one plane for
    stone positions of each color and one for ko.
    '''

    def __init__(self, data_directory='data', num_planes=3, consolidate=True, use_generator=False):
        super(ThreePlaneProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)

    def feature_and_label(self, color, move, go_board):
        '''
        Parameters
        ----------
        color: color of the next person to move
        move: move they decided to make
        go_board: represents the state of the board before they moved

        Planes we write:
        0: our stones
        1: their stones
        2: ko
        '''
        row, col = move
        enemy_color = go_board.other_color(color)
        move_array = np.zeros((self.num_planes, go_board.board_size, go_board.board_size))
        for row in range(0, go_board.board_size):
            for col in range(0, go_board.board_size):
                pos = (row, col)
                if go_board.board.get(pos) == color:
                    move_array[0, row, col] = 1
                if go_board.board.get(pos) == enemy_color:
                    move_array[1, row, col] = 1
                if go_board.is_simple_ko(color, pos):
                    move_array[2, row, col] = 1
        return move_array, move

