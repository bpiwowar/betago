import subprocess
import re
import argparse
import logging

from betago.model import GoModel
from betago.processor import SevenPlaneProcessor
from betago.gtp.board import gtp_position_to_coords, coords_to_gtp_position
from betago.dataloader.goboard import GoBoard


letters = 'abcdefghijklmnopqrs'

def sgfCoord(coords):
    row, col = coords
    return letters[col] + letters[18 - row]


class Model(GoModel):
    def __init__(self, *args):
        super().__init__(None, None)
        # self.command = ["tee", "/tmp/a"] #
        self.command = ["gnugo", "--mode", "gtp"]
        self.p = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        self.send_command("boardsize %d\n" % self.go_board.board_size)
        self.get_response()

    def send_command(self, cmd):
        self.p.stdin.write(cmd.encode("ascii"))
        self.p.stdin.flush()

    def get_response(self):
        succeeded = False
        result = ''
        while succeeded == False:
            line = self.p.stdout.readline().decode('ascii')
            if line[0] == '=':
                succeeded = True
                line = line.strip()
                logging.info("Response is: %s", line)
                result = re.sub('^= ?', '', line)
        return result


    def set_board(self, board: GoBoard):
        '''Set the board to a specific state.'''
        self.board = board

        self.send_command(b"boardsize %d\n" % board.boardsize)
        self.get_response()

        self.send_command(b"komi 7.5\n")
        self.get_response()

        # if(handicap == 0):
        #     self.send_command("komi 7.5\n")
        #     get_response(p)
        #     sgf = sgf + "KM[7.5]\n"
        # else:
        #     send_command(p, "fixed_handicap " + str(handicap) + "\n")
        #     stones = get_response(p)
        #     sgf_handicap = "HA[" + str(handicap) + "]AB"
        #     for pos in stones.split(" "):
        #         move = gtp_position_to_coords(pos)
        #         bot.apply_move('b', move)
        #         sgf_handicap = sgf_handicap + "[" + sgfCoord(move) + "]"
        #     sgf = sgf + sgf_handicap + "\n"

    def apply_move(self, color, move):
        ''' Apply the human move'''
        if move is None:
            self.send_command("play " + color + " pass\n")
        else:
            pos = coords_to_gtp_position(move)
            self.send_command("play " + color + " " + pos + "\n")
        
        resp = self.get_response()

    def select_move(self, bot_color):
        ''' Select a move for the bot'''
        self.send_command("genmove " + bot_color + "\n")
        pos = self.get_response()
        if pos == 'RESIGN' or pos == 'PASS':
            return None
        else:
            return gtp_position_to_coords(pos)
