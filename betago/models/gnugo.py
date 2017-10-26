from __future__ import print_function
import yaml
import subprocess
import re
import argparse

from betago.model import GoModel
from betago.processor import SevenPlaneProcessor
from betago.gtp.board import gtp_position_to_coords, coords_to_gtp_position


letters = 'abcdefghijklmnopqrs'

def sgfCoord(coords):
    row, col = coords
    return letters[col] + letters[18 - row]


class Model(GoModel):
    def __init__(self):
        super().init(None, None)

        self.gnugo_cmd = ["gnugo", "--mode", "gtp"]
        self.p = subprocess.Popen(gnugo_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        # deal with handicap.  Parse multi-stone response to see where it was placed.
        # handicap = args.handicap[0]

        self.sendcommand("boardsize 19\n")
        self.get_response()

        sgf = "(;GM[1]FF[4]CA[UTF-8]SZ[19]RU[Chinese]\n"

        if handicap == 0:
            self.sendcommand("komi 7.5\n")
            self.get_response()
            sgf = sgf + "KM[7.5]\n"
        else:
            self.sendcommand("fixed_handicap " + str(handicap) + "\n")
            stones = self.get_response()
            sgf_handicap = "HA[" + str(handicap) + "]AB"
            for pos in stones.split(" "):
                move = gtp_position_to_coords(pos)
                bot.apply_move('b', move)
                sgf_handicap = sgf_handicap + "[" + sgfCoord(move) + "]"
            sgf = sgf + sgf_handicap + "\n"

    def self.sendcommand(self, cmd):
        self.p.stdin.write(cmd)
        print(cmd.strip())

    def get_response(self):
        succeeded = False
        result = ''
        while succeeded == False:
            line = self.p.stdout.readline()
            if line[0] == '=':
                succeeded = True
                line = line.strip()
                print("Response is: " + line)
                result = re.sub('^= ?', '', line)
        return result


    def set_board(self, board):
        '''Set the board to a specific state.'''
        self.send_command("boardsize %d\n" % board.boardsize)
        self.get_response()

        sgf = "(;GM[1]FF[4]CA[UTF-8]SZ[19]RU[Chinese]\n"

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
            self.sendcommand("play " + colors[our_color] + " pass\n")
            sgf = sgf + ";" + our_color.upper() + "[]\n"
            passes = passes + 1
        else:
            pos = coords_to_gtp_position(move)
            self.sendcommand("play " + colors[our_color] + " " + pos + "\n")
            sgf = sgf + ";" + our_color.upper() + "[" + sgfCoord(move) + "]\n"
            passes = 0
        
        resp = get_response(p)

    def select_move(self, bot_color):
        ''' Select a move for the bot'''
        self.sendcommand("genmove " + colors[their_color] + "\n")
        pos = get_response(p)
        if(pos == 'RESIGN'):
            passes = 2
        elif(pos == 'PASS'):
            sgf = sgf + ";" + their_color.upper() + "[]\n"
            passes = passes + 1
        else:
            move = gtp_position_to_coords(pos)
            bot.apply_move(their_color, move)
            sgf = sgf + ";" + their_color.upper() + "[" + sgfCoord(move) + "]\n"
            passes = 0
