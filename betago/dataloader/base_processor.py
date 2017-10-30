# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
import json
import struct
import os
import io
import gc
import glob
import os.path
import tarfile
import gzip
import shutil
import numpy as np
import argparse
import multiprocessing
from os import sys
from queue import Queue
import logging
from pathlib import Path
import traceback as tb

from .. import gosgf
from .goboard import GoBoard
from .index_processor import KGSIndex
from .sampling import Sampler

def worker(path):
    try:
        worker.processor.process_zip(path, worker.queue, worker.sampler, worker.boardsize)
        worker.queue.put(None)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')
    except Exception as e:
        tb.print_exc()
        worker.queue.put(e)

def worker_init(processor, sampler, queue, boardsize):
    worker.processor = processor
    worker.queue = queue
    worker.sampler = sampler
    worker.boardsize = boardsize

class DataGenerator(object):
    def __init__(self, data_dir, samples):
        self.data_dir = data_dir
        self.files = set(file_name for file_name, index in samples)
        self.samples = samples
        self.num_samples = None

    def get_num_samples(self, batch_size=128, nb_classes=19 * 19):
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size, nb_classes=nb_classes):
                self.num_samples += X.shape[0]
            return self.num_samples

    def _generate(self, batch_size, nb_classes):
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = self.data_dir + '/' + file_name + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                X = np.load(feature_file)
                y = np.load(label_file)
                X = X.astype('float32')
                y = np_utils.to_categorical(y.astype(int), nb_classes)
                gc.collect()
                while X.shape[0] >= batch_size:
                    X_batch, X = X[:batch_size], X[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    gc.collect()
                    yield X_batch, y_batch
            gc.collect()

    def generate(self, batch_size=128, nb_classes=19 * 19):
        while True:
            for item in self._generate(batch_size=batch_size, nb_classes=nb_classes):
                yield item


class GoBaseProcessor(object):
    '''
    Abstract base class for Go data processing. To implement this class, implement
    process_zip and consolidate_games below.

    Generally speaking, processing load is split between available CPUs, thereby generating intermediate
    files for each worker. Processing many files can produce massive amounts of data that will not (easily)
    fit into memory. To avoid overflow, deactivate file consolidation by initializing with consolidate=False.
    '''

    def __init__(self, data_directory='data', num_planes=7):
        '''
        Parameters:
        -----------
        data_dir: relative path to store data files, defaults to 'data'
        num_planes: Number of Go board copies used as input, corresponds to the number of input channels
                    in a CNN.
        consolidate: Boolean flag to indicate if intermediate results should be consolidated into one, which
                     can be very expensive.
        '''
        self.data_dir = data_directory
        self.num_planes = num_planes

    def prepare_go_data(self, kgs: KGSIndex, datadir: Path, sampler: Sampler, *,
        cores=None, buffer_size=25000, boardsize=19):
        '''
        Main method to load go data.

        Loads and initializes an index from KGS and downloads zipped files. Depending on provided type, unzip and
        process data, then optionally store it in one consolidated file.

        Parameters:
        -----------
        index: a KGS index
        '''
        self.map_to_workers(kgs.data_directory, datadir, sampler, buffer_size, boardsize=boardsize, cores=cores)
        logging.info('Finished processing')

    def get_handicap(self, go_board, sgf):
        ''' Get handicap stones '''
        first_move_done = False
        if sgf.get_handicap() != None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    go_board.apply_move('b', move)
            first_move_done = True
        return go_board, first_move_done

    def to_json(self):
        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "parameters": self.__dict__
        }

    @staticmethod
    def load(v):
        import importlib
        m = importlib.import_module(v["module"])
        s = getattr(m, v["class"])()
        for key, value in v["parameters"].items():
            setattr(s, key, value)
        return s

    def map_to_workers(self, kgsdir, datadir: Path, sampler, buffer_size, *, cores=None, boardsize=19):
        '''
        Determine the list of zip files that need to be processed, then map load
        to number of available CPUs.
        '''
        from .data import BinaryRepresentation
        
        # Transform the above dictionary to a list that can be processed in parallel
        cores = multiprocessing.cpu_count() if not cores else cores
        manager = multiprocessing.Manager()
        queue = manager.Queue(maxsize=2*cores)
        
        to_process = []
        for path in Path(kgsdir).iterdir():
            if path.name.endswith('.tar.gz'):
                to_process.append(path)

        processor_json = self.to_json()
        print(json.dumps(processor_json))

        # Launch processes
        logging.info("Starting processing the KGS files [%d threads]", cores)
        pool = multiprocessing.Pool(cores, worker_init, [self, sampler, queue, boardsize])
        p = pool.map_async(worker, to_process)
        buffer = []
        counts = [0, 0, 0]
        count = len(to_process)
        with open(datadir.joinpath("train.dat"), "wb") as train_fh, open(datadir.joinpath("validation.dat"), "wb") as val_fh, open(datadir.joinpath("test.dat"), "wb") as test_fh:
            try:
                fhs = [train_fh, val_fh, test_fh]
                def process(record):
                    '''Write record to disk'''
                    fhs[record[0]].write(record[1])
                    counts[record[0]] += 1

                while True:
                    v = queue.get()
                    if v is None:
                        # Finished processing one item
                        count -= 1
                        logging.info("Worker finished: %d tasks remaining", count)
                        if count == 0: 
                            break


                    elif isinstance(v, Exception):
                        logging.error("Detected exception in thread - aborting...")
                        pool.terminate()
                        pool.join()
                        raise v
                        
                    elif len(buffer) < buffer_size:
                        buffer.append(v)
                        if len(buffer) == buffer_size:
                            logging.info("Buffer full")
                    else:
                        ix = sampler.random.randrange(len(buffer))
                        process(buffer[ix])
                        buffer[ix] = v

                np.random.shuffle(buffer)
                for v in buffer:
                    process(v)
                logging.info("Finishing writing: %s", counts)

                for fh in fhs:
                    fh.write(BinaryRepresentation.END)

                with open(datadir.joinpath("information.json"), "w") as fp:
                    json.dump({
                        "train": counts[Sampler.TRAIN],
                        "validation": counts[Sampler.VALIDATION],
                        "test": counts[Sampler.TEST],
                        
                        "boardsize": boardsize,
                        "numplanes": self.num_planes,
                        "processor": processor_json
                    }, fp)
                        
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
                pool.join()
                sys.exit(-1)

    def init_go_board(self, sgf_contents):
        ''' Initialize a go board from SGF file content'''
        sgf = gosgf.Sgf_game.from_string(sgf_contents)
        return sgf, GoBoard(19)


class GoDataProcessor(GoBaseProcessor):
    '''
    GoDataProcessor generates data, e.g. numpy arrays, of features and labels and returns them to the user.
    '''
    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(GoDataProcessor, self).__init__(data_directory=data_directory,
                                              num_planes=num_planes)
        self.use_generator = use_generator

    def feature_and_label(self, color, move, go_board):
        '''
        Given a go board and the next move for a given color, treat the next move as label and
        process the current board situation as feature to learn this label.

        return: X, y - feature and label
        '''
        return NotImplemented
     
    def process_zip(self, path, queue, sampler, boardsize):
        from .data import BitStream, BinaryRepresentation

        # Read zipped file and extract name list
        logging.info("Processing file %s", path)
        
        # Write body and close file
        with tarfile.open(path, "r:*") as tf:
            for entry in tf:  # list each entry one by one
                if not entry.name.endswith(".sgf"):
                    continue
                # logging.info("Processing %s", entry.name)
                mode = sampler.game()
                if mode == Sampler.IGNORE:
                    continue

                with tf.extractfile(entry) as fh:
                    sgf_content = fh.read()
                sgf, go_board_no_handy = self.init_go_board(sgf_content)
                go_board, first_move_done = self.get_handicap(go_board_no_handy, sgf)

                # Ignore
                if go_board.board_size != boardsize:
                    continue

                # Contain the size of each board
                for item in sgf.main_sequence_iter():
                    (color, move) = item.get_move()
                    if color is not None and move is not None:
                        row, col = move
                        if first_move_done:
                            if sampler.sample_board(mode):
                                buffer = io.BytesIO()
                                bs = BitStream(buffer)
                                matrix, _ = self.feature_and_label(color, move, go_board)
                                b = BinaryRepresentation.compress(bs, matrix, move)
                                bs.flush()
                                queue.put([mode, buffer.getvalue()])

                        first_move_done = True
                        go_board.apply_move(color, (row, col))

            logging.info("Finished processing %s", path)
