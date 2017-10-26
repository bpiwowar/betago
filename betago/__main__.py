# Main file

import argparse
import logging
import sys
import pathlib
import importlib
from pathlib import Path
import re

from .dataloader.data import Dataset
from .dataloader.base_processor import GoBaseProcessor
from .dataloader.sampling import SingleSampler, Sampler
from .commands import Command, command, argument, commands

logging.basicConfig(level=logging.INFO)


# ---- Go Games data pre-processing

@command(description="Download KGS files")
def kgs(cfg, args):
    from .dataloader.index_processor import KGSIndex
    index = KGSIndex(data_directory=str(cfg.kgsdirectory), index_page=str(cfg.kgsdirectory.joinpath('kgs_index.html')))
    logging.info("Downloading data files from KGS server")
    index.download_files()

@argument("--processor", required=True, choices=["SevenPlane", "ThreePlane"])
@argument("--boardsize", default=19, help="Board size of processed games")
@argument("--cores", required=False, type=int, default=0, help="Number of cores to use (default to number of available cores)")
@argument("--seed", required=False, type=int, default=0, help="Seed for random generator")

@argument("--train", required=False, type=SingleSampler, default=SingleSampler("10::10"), help="Ratio of games to use for training")
@argument("--validation", required=False, type=SingleSampler, default=SingleSampler("10::10"), help="Ratio of games to use for validation")
@argument("--test", required=False, type=SingleSampler, default=SingleSampler("10::10"), help="Ratio of games to use for training")

@argument("--buffer-size", required=False, type=int, default=25000, help="Buffer for games")
@command(description="Prepare KGS files for training")
def prepare(cfg, args):
    print(repr(args.train))

    if args.train.game_ratio+args.validation.game_ratio+args.test.game_ratio > 1:
        raise Exception("Sum of train/val/test ratio cannot be greater than 1")

    from .dataloader.index_processor import KGSIndex
    import importlib

    processors = importlib.import_module(".processor", __package__)
    processorclass = getattr(processors, args.processor + "Processor")

    index = KGSIndex(data_directory=str(cfg.kgsdirectory), index_page=str(cfg.kgsdirectory.joinpath('kgs_index.html')))

    dirname = args.processor
    dirname += '%d-%d-train-%s-val-%s-test-%s' % (args.seed, args.boardsize, args.train, args.validation, args.test)
    datadir = cfg.datadirectory.joinpath(dirname)

    logging.info("Outputs in directory %s", datadir)
    datadir.mkdir(exist_ok=True)
    if datadir.joinpath("information.json").file_exists():
        raise Exception("information.json file exist - dataset already pre-processed")

    processor = processorclass()
    sampler = Sampler(args.train, args.validation, args.test, seed=0)
    processor.prepare_go_data(index, datadir, sampler,
        cores=args.cores, buffer_size=args.buffer_size, boardsize=args.boardsize)


@argument("boardspath", type=Path)
@argument("type", choices=["train", "validation", "test"])
@argument("board", type=int, help="Index of the board (otherwise outputs everything)")
@command(description="Inspect a compressed train file")
def show(cfg, args):
    from .dataloader.base_processor import BinaryRepresentation
    from .dataloader.data import BitStream, Dataset

    dataset = Dataset(args.boardspath)
    with dataset[args.type] as data:
        print(data[args.board])
    
        data.readall()


# --- Model training


@argument("data",  type=Path, help="Directory containing the training data")
@argument("model", help="Name of the model (corresponds to a python module)")
@argument("parameters", type=Path, help="Directory containing the model parameters")
@argument("--batchsize", type=int, default=10000, help="Batch size (train)")
@argument("--checkpoint", type=int, default=100, help="Number of training iterations before checkpoint")
@argument("--reset", default=False, action="store_true", help="Reset the model")
@command(description="Supervised training of a policy model")
def direct_policy_train(cfg, args):
    ds = Dataset(args.data)

    m = importlib.import_module(args.model)
    model = m.Model(args.arguments, args.parameters)

    if not args.reset and not model.restore():
        model.init(ds.processor, ds.boardsize)

    with ds["test"] as data:
        logging.info("Loading test data")
        test_data = data.readall()
    # with ds["validation"] as data:
    #     logging.info("Loading validation data")
    #     validation_data = data.readall()

    with ds["train"] as data:
        for train_data in data.batches(args.batchsize):
            logging.info("Training...")
            train_cost = model.train(*train_data)
            test_cost = model.cost(*test_data)
            if (model.epoch % args.checkpoint) == 0:
                model.save_checkpoint()
            logging.info("Iteration %d: train=%g, test=%g", model.epoch, train_cost, test_cost)

# --- Model testing

def getmodel(model, parameters, load=True):
    from .model import ModelBot, GoModel

    m = importlib.import_module(model)
    if issubclass(m.Model, GoModel):
        return m.Model()

    model = m.Model(args.parameters)
    if load:
        model.load()
    return ModelBot(model=model, processor=model.processor)



@argument('--host', default='localhost', help='host to listen to')
@argument('--port', '-p', type=int, default=8080, help='Port the web server should listen on (default 8080).')
@argument("model", help="Name of the model (corresponds to a python module)")
@argument("parameters", type=Path, help="Directory containing the model parameters")
@command(description='Start a GO server and opens a web page')
def demo(cfg, args):
    from .model import ModelBot, HTTPFrontend

    m = importlib.import_module(args.model)
    bot = getmodel(args.model, args.parameters)

    go_server = HTTPFrontend(bot=bot, port=args.port)
    go_server.run()


@argument('--host', default='localhost', help='host to listen to')
@argument('--port', '-p', type=int, default=8080, help='Port the web server should listen on (default 8080).')
@argument("model", help="Name of the model (corresponds to a python module)")
@argument("parameters", type=Path, help="Directory containing the model parameters")
@command(description='Start a GTP server')
def gtp(cfg, args):
    from .model import ModelBot, HTTPFrontend
    from .gtp import GTPFrontend

    model = getmodel(args.model, args.parameters)

    logging.info("Starting GTP frontend")
    frontend = GTPFrontend(bot=ModelBot(model))
    frontend.run()



@argument("model1", help="Name of the model (corresponds to a python module)")
@argument("parameters1", type=Path, help="Directory containing the model parameters")
@argument("model2", help="Name of the model (corresponds to a python module)")
@argument("parameters2", type=Path, help="Directory containing the model parameters")
@argument('--komi', '-k', type=float, default=5.5)
@command(description='Simulate a game between two bots')
def simulate(cfg, args):
    from .model import ModelBot, HTTPFrontend
    from .dataloader import goboard
    import betago.scoring as scoring
    import betago.simulate as simulate

    # Load models
    black_bot = getmodel(args.model1, args.parameters1)
    white_bot = getmodel(args.model2, args.parameters2)

    # Simulate
    board = goboard.GoBoard()
    simulate.simulate_game(board, black_bot, white_bot)
    
    print("Game over!")
    print(goboard.to_string(board))
    # Does not remove dead stones.
    print("\nScore (Chinese rules):")
    status = scoring.evaluate_territory(board)
    black_area = status.num_black_territory + status.num_black_stones
    white_area = status.num_white_territory + status.num_white_stones
    white_score = white_area + args.komi
    print("Black %d" % black_area)
    print("White %d + %.1f = %.1f" % (white_area, args.komi, white_score))

# --- Parse command line

class Configuration:
    def __init__(self, datadir=None):
        self.datadir = pathlib.Path(datadir)

    @property
    def kgsdirectory(self):
        return self.datadir.joinpath("kgs")

    @property
    def datadirectory(self):
        return self.datadir.joinpath("processed")
        
parser = argparse.ArgumentParser(description='betago')

parser.add_argument("--verbose", action="store_true", help="Be verbose")
parser.add_argument("--debug", action="store_true", help="Be even more verbose (implies traceback)")
parser.add_argument("--profile", help="Output profile file (turns profiling on)")
parser.add_argument("--data", help="Data directory (default: data of the current directory)", default=str(pathlib.Path.cwd().joinpath("data")))

parser.add_argument("command", choices=commands.keys())
parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments for the preparation")

args = parser.parse_args()

if args.command is None:
    parser.print_help()
    sys.exit()

if args.verbose:
    logging.getLogger().setLevel(logging.INFO)

if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)

try:
    cfg = Configuration(datadir=args.data)
    if args.profile:
        logging.info("Profiling on - output file %s", args.profile)
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    commands[args.command](cfg, args)
    if args.profile:
        pr.dump_stats(args.profile)
    
except Exception as e:
    sys.stderr.write("Error while running command %s:\n" % args.command)
    sys.stderr.write(str(e))

    import traceback
    sys.stderr.write(traceback.format_exc())

    sys.exit(1)
