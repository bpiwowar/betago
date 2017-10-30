from __future__ import absolute_import
from __future__ import print_function
import copy
import random
import pickle
from itertools import chain, product
from multiprocessing import Process

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

import os.path as op
import logging
import re
import cgi
import json
import inspect
from pathlib import Path
import numpy as np
from . import scoring
from .dataloader.goboard import GoBoard
from .dataloader.base_processor import GoBaseProcessor
from .processor import ThreePlaneProcessor
from six.moves import range


def routes(clazz):
    clazz.__routes__ = {}
    for m in dir(clazz):
        m = getattr(clazz, m)
        route = getattr(m, "__route__", None)
        if route:
            clazz.__routes__[route] = m
    logging.info("Detected routes: %s", clazz.__routes__.keys())
    return clazz

class route:
    def __init__(self, path, methods=None):
        self.pathre = re.compile("^%s$" % path)
        self.methods = methods
    def __call__(self, method):
        method.__route__ = self
        self.method = method
        self.parameters = inspect.signature(self.method).parameters
        return method

    def __repr__(self):
        return "route(%s)" % self.pathre

    def send_headers(self, http, size=None):
        ctype = "text/html"
        http.send_response(HTTPStatus.OK)
        http.send_header("Content-type", ctype)
        if size:
            http.send_header("Content-Length",size)
        # self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        http.end_headers()

    def process(self, http):
        m = self.pathre.match(http.path)
        logging.debug("Matching %s with %s [%s]...", http.path, self.pathre, not(m is None))
        if not m: 
            return False

        logging.info("Processing with %s [%s]...", self.method.__name__, m.groupdict())

        values = m.groupdict()
        args = []
        kwargs = {}
        first = True
        for k, v in self.parameters.items():
            if first:
                first = False
                args.append(http)
            elif v.kind == inspect.Parameter.POSITIONAL_ONLY or v.default == inspect.Signature.empty:
                    args.append(values[k])
            else:
                kwargs[k] = values.get(k, v.default)

        logging.debug("Calling %s with %s and %s", self.method, args, kwargs)
        output = self.method(*args, **kwargs)
        self.send_headers(http, size=len(output))
        http.wfile.write(output)
        return True



class HTTPFrontend():
    '''
    HTTPFrontend is a simple http server localhost:8080, exposing a REST API to predict
    go moves.
    '''

    def __init__(self, bot, port=8080):
        self.bot = bot
        self.port = port

    def start_server(self):
        ''' Start Go model server '''
        self.server = Process(target=self.start_service)
        self.server.start()

    def stop_server(self):
        ''' Terminate Go model server '''
        self.server.terminate()
        self.server.join()

    def run(self):
        ''' Run app'''    
        @routes
        class HTTPRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                for route, method in HTTPRequestHandler.__routes__.items():
                    if route.process(self):
                        return
                logging.error("Could not process %s", self.path)

                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()

            def do_POST(self):
                ctype, pdict = cgi.parse_header(self.headers['content-type'])
                logging.debug("Handling post of content-type: %s", ctype)
                if ctype == 'application/json':
                    logging.debug("File is %s / %s", self.rfile, self.headers["content-length"])
                    s = self.rfile.read(int(self.headers["content-length"]))
                    self.json = json.loads(s)
                    logging.debug("JSON payload: %s", self.json)
                else:
                    raise Exception("Cannot handle content-type %s", ctype)

                return self.do_GET()

            @route(r'/dist/(?P<path>.*)')
            def static_file_dist(request, path):
                return open("ui/dist/" + path, "rb").read()

            @route('/large/(?P<path>.*)')
            def static_file_large(request, path):
                return open("ui/large/" + path, "rb").read()

            @route('/')
            def home(request):
                # Inject game data into HTML
                board_init = 'initialBoard = ""' # backup variable
                board = {}
                for row in range(19):
                    board_row = {}
                    for col in range(19):
                        # Get the cell value
                        cell = str(self.bot.go_board.board.get((col, row)))
                        # Replace values with numbers
                        # Value will be be 'w' 'b' or None
                        cell = cell.replace("None", "0")
                        cell = cell.replace("b", "1")
                        cell = cell.replace("w", "2")
                        # Add cell to row
                        board_row[col] = int(cell) # must be an int
                    # Add row to board
                    board[row] = board_row
                board_init = str(board).encode("utf-8") # lazy convert list to JSON
                
                # output the modified HTML file
                return open("ui/demoBot.html", "rb").read().replace(b'"__i__"', b'var boardInit = ' + board_init) 

            @route('/sync', methods=['GET', 'POST'])
            def exportJSON(request):
                export = {}
                export["hello"] = "yes?"
                return json.dumps(export).encode("utf-8")

            @route('/prediction', methods=['GET', 'POST'])
            def next_move(request):
                '''Predict next move and send to client.

                Parses the move and hands the work off to the bot.
                '''
                content = request.json
                col = content['i']
                row = content['j']
                logging.info('Received move: %s', (row, col))
                self.bot.apply_move('b', (row, col))

                bot_row, bot_col = self.bot.select_move('w')
                logging.info('Predicted move: %s', (bot_col, bot_row))
                result = {'i': bot_col, 'j': bot_row}
                json_result = json.dumps(result)
                return json_result.encode("utf-8")
        
        
        server_address = ('' if op.isfile("/.dockerenv") else "127.0.0.1", self.port)
        httpd = HTTPServer(server_address, HTTPRequestHandler)
        logging.info("Running server on port %d", self.port)
        httpd.serve_forever()



class GoModel(object):
    '''Tracks a board and selects moves.'''

    def __init__(self, model):
        '''
        Parameters:
        -----------
        processor: Instance of gammago.processor.GoDataLoader, e.g. SevenPlaneProcessor
        model: In principle this can be anything that can predict go moves, given data provided by the above
               processor. In practice it may very well be (an extension of) a keras model plus glue code.
        '''
        self.go_board = GoBoard(19)
        self.model = model

    def set_board(self, board):
        '''Set the board to a specific state.'''
        self.go_board = copy.deepcopy(board)

    def apply_move(self, color, move):
        ''' Apply the human move'''
        return NotImplemented

    def select_move(self, bot_color):
        ''' Select a move for the bot'''
        return NotImplemented



def get_first_valid_move(board, color, move_generator):
    for move in move_generator:
        if move is None or board.is_move_legal(color, move):
            return move
    return None

def generate_in_random_order(point_list):
    """Yield all points in the list in a random order."""
    point_list = copy.copy(point_list)
    random.shuffle(point_list)
    for candidate in point_list:
        yield candidate


def all_empty_points(board):
    """Return all empty positions on the board."""
    empty_points = []
    for point in product(list(range(board.board_size)), list(range(board.board_size))):
        if point not in board.board:
            empty_points.append(point)
    return empty_points


def fill_dame(board):
    status = scoring.evaluate_territory(board)
    # Pass when all dame are filled.
    if status.num_dame == 0:
        yield None
    for dame_point in generate_in_random_order(status.dame_points):
        yield dame_point



from gammago.commands import argument

class BaseModel:
    '''Base model for all learned models'''

    def load(self):
        pass
        
try:
    import torch

    # Move to model
    class TorchModel(torch.nn.Module, BaseModel):
        def __init__(self):
            BaseModel.__init__(self)
            torch.nn.Module.__init__(self)
            self.epoch = 0
            self.optimizer = None
            # List of parameters to (de)serialize
            self.__parameters__ = []

        def init(self, processor: GoBaseProcessor, boardsize: int, state):
            self.processor = processor
            self.numplanes = processor.num_planes
            self.boardsize = boardsize
            self.construct()

        def load(self, path):
            with open(path, 'rb') as fp:
                state = torch.load(fp)

            processor = GoBaseProcessor.load(state["processor"])
            self.init(processor, state["boardsize"], state)
            self.epoch = state["epoch"]
            self.load_state_dict(state["state_dict"])
            self.optimizer.load_state_dict(state["optimizer"])

        def restore(self, path: Path):            
            if path.is_file():
                self.load(path)
                logging.info("Restored model (epoch %d)", self.epoch)
                return True
            return False

        def save_checkpoint(self, path: Path):
            state = {
                'epoch': self.epoch + 1,
                'state_dict': self.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'processor': self.processor.to_json(),
                'boardsize': self.boardsize,
                'module': self.__class__.__module__,
                'class': self.__class__.__name__
            }   

            for key in self.__parameters__:
                state[key] = getattr(self, key)

            
            tmpfilepath = path.with_suffix(".tmp")
            with open(tmpfilepath, "wb") as fp:
                torch.save(state, fp)

            tmpfilepath.replace(path)

        def construct(self):
            raise NotImplemented
        def train(self, boards, labels):
            raise NotImplemented
        def cost(self, boards, labels):
            raise NotImplemented
        def predict(self, board):
            raise NotImplemented

except ImportError as e:
    if e.name == 'torch':
        logging.debug("torch is not installed: skipping torch model [%s]", e.name)
    raise

class ModelBot(GoModel):
    '''
    ModelBot takes top_n predictions of a model and tries to apply the best move. If that move is illegal,
    choose the next best, until the list is exhausted. If no more moves are left to play, continue with random
    moves until a legal move is found.
    '''

    def __init__(self, model, top_n=10):
        super(ModelBot, self).__init__(model=model)
        self.top_n = top_n
        self.processor = model.processor

    def apply_move(self, color, move):
        # Apply human move
        self.go_board.apply_move(color, move)

    def select_move(self, bot_color):
        move = get_first_valid_move(self.go_board, bot_color,
                                    self._move_generator(bot_color))
        if move is not None:
            self.go_board.apply_move(bot_color, move)
        return move

    def _move_generator(self, bot_color):
        return chain(
            # First try the model.
            self._model_moves(bot_color),
            # If none of the model moves are valid, fill in a random
            # dame point. This is probably not a very good move, but
            # it's better than randomly filling in our own eyes.
            fill_dame(self.go_board),
            # Lastly just try any open space.
            generate_in_random_order(all_empty_points(self.go_board)),
        )

    def _model_moves(self, bot_color):
        # Turn the board into a feature vector.
        # The (0, 0) is for generating the label, which we ignore.
        X, label = self.processor.feature_and_label(
            bot_color, (0, 0), self.go_board)
        X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))

        # Generate bot move.
        pred = np.squeeze(self.model.predict(X))
        top_n_pred_idx = pred.argsort()[-self.top_n:][::-1]
        for idx in top_n_pred_idx:
            prediction = int(idx)
            pred_row = prediction // 19
            pred_col = prediction % 19
            pred_move = (pred_row, pred_col)
            yield pred_move


class RandomizedModelBot(GoModel):
    '''
    Takes a weighted sample from the predictions of a keras model. If none of those moves is legal,
    pick a random move.
    '''

    def __init__(self, model, processor):
        super(RandomizedModelBot, self).__init__(model=model, processor=processor)

    def apply_move(self, color, move):
        # Apply human move
        self.go_board.apply_move(color, move)

    def select_move(self, bot_color):
        move = get_first_valid_move(self.go_board, bot_color,
                                    self._move_generator(bot_color))
        if move is not None:
            self.go_board.apply_move(bot_color, move)
        return move

    def _move_generator(self, bot_color):
        return chain(
            # First try the model.
            self._model_moves(bot_color),
            # If none of the model moves are valid, fill in a random
            # dame point. This is probably not a very good move, but
            # it's better than randomly filling in our own eyes.
            fill_dame(self.go_board),
            # Lastly just try any open space.
            generate_in_random_order(all_empty_points(self.go_board)),
        )

    def _model_moves(self, bot_color):
        # Turn the board into a feature vector.
        # The (0, 0) is for generating the label, which we ignore.
        X, label = self.processor.feature_and_label(
            bot_color, (0, 0), self.go_board, self.num_planes)
        X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))

        # Generate moves from the keras model.
        n_samples = 20
        pred = np.squeeze(self.model.predict(X))
        # Cube the predictions to increase the difference between the
        # best and worst moves. Otherwise, it will make too many
        # nonsense moves. (There's no scientific basis for this, it's
        # just an ad-hoc adjustment)
        pred = pred * pred * pred
        pred /= pred.sum()
        moves = np.random.choice(19 * 19, size=n_samples, replace=False, p=pred)
        for prediction in moves:
            pred_row = prediction // 19
            pred_col = prediction % 19
            yield (pred_row, pred_col)

