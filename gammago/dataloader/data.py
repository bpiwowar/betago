from pathlib import Path
import struct
import json
import logging
import numpy as np
import gzip
import threading
import queue
import traceback
from .base_processor import GoBaseProcessor

class BitStream:
    Values = [128, 64, 32, 16, 8, 4, 2, 1]

    def __init__(self, fh):
        self.fh = fh

        # Current bit position and byte value
        self.bitpos = 0
        self.byte = 0

    def tell(self):
        return self.fh.tell() * 8 + self.bitpos

    def readbit(self):
        # Read next byte
        if self.bitpos == 0:
            self.byte = self.fh.read(1)[0]
        
        # Get value
        v = 1 if self.byte & BitStream.Values[self.bitpos] else 0

        # Increment bit position
        self.bitpos += 1
        if self.bitpos == 8:
            self.bitpos = 0

        return v
    
    def readgamma(self):
        n = 0
        while self.readbit():
            n += 1
        x = 1
        for i in range(n):
            x <<= 1
            b = self.readbit()
            x += b
        return x

    def writebit(self, v: int):
        if v:
            self.byte |= BitStream.Values[self.bitpos]

        self.bitpos += 1
        if self.bitpos == 8:
            self.fh.write(bytes([self.byte]))
            self.bitpos = 0
            self.byte = 0

    def write(self, x):
        if self.bitpos != 0:
            raise Exception("Not aligned bit position")
        self.fh.write(x)

    def read(self, x):
        if self.bitpos != 0:
            raise Exception("Not aligned bit position")
        return self.fh.read(x)

    def writegamma(self, x):
        assert x >= 1

        # Write exponent (unary)
        y = x
        y >>= 1
        m = 1
        while y > 0:
            self.writebit(1)
            y >>= 1
            m <<= 1
        self.writebit(0)

        # Write 
        m >>= 1
        while m > 0:
            b = 1 if (x & m) else 0
            self.writebit(b)
            m >>= 1

    def align(self):
        self.bitpos = 0

    def flush(self):
        if self.bitpos > 0:
            self.fh.write(bytes([self.byte]))
            self.byte = 0
            self.bitpos = 0


def decompose(v, bases):
    """Decompose an integer according to a variable basis"""
    t = []
    for b in bases:
        t.append(v % b)
        v //= b
    return t

class BinaryRepresentation:
    """Compact binary representation of binary tensors"""
    
    # HEADER
    POSITION_HEADER = b'GO'
    END = b'__END__'

    # fields = total_examples, board_size, board_size, num_planes, bits_per_pixel
    LABELSTRUCT = struct.Struct("II")
    SIZESTRUCT = struct.Struct("I")

    def __init__(self, filename):
        self.filename = filename

    def read(bs: BitStream, num_planes, boardsize):
        v = bs.read(len(BinaryRepresentation.POSITION_HEADER))
        if v != BinaryRepresentation.POSITION_HEADER:
            raise Exception("Board header was not found")
        
        v = bs.read(BinaryRepresentation.LABELSTRUCT.size)
        move = BinaryRepresentation.LABELSTRUCT.unpack(v)

        v = bs.read(BinaryRepresentation.SIZESTRUCT.size)
        n, = BinaryRepresentation.SIZESTRUCT.unpack(v)

        # Did we encode ones or zeros?
        mode = 1 if bs.readbit() else 0
        a = (np.zeros if mode else np.ones)((num_planes, boardsize, boardsize))

        # Decode the matrix
        p = -1
        shape = (a.shape[0], a.shape[2], a.shape[1])
        for i in range(n):
            p += bs.readgamma()
            plane, col, row = decompose(p, shape)
            a[plane, row, col] = mode

        bs.align()
        return a, move

    @staticmethod
    def compress(bs, matrix, move):            
        """
        """
        bs.write(BinaryRepresentation.POSITION_HEADER)
        bs.write(BinaryRepresentation.LABELSTRUCT.pack(*move))

        s = int(matrix.sum())
        if s > matrix.size / 2:
            b = 0
            bs.write(BinaryRepresentation.SIZESTRUCT.pack(matrix.size - s))
        else:
            b = 1
            bs.write(BinaryRepresentation.SIZESTRUCT.pack(s))

        bs.writebit(1 if b == 0 else 0)
        d = 1
        for row in range(matrix.shape[1]):
            for col in range(matrix.shape[2]):
                for plane in range(matrix.shape[0]):
                    if matrix[plane, row, col] == b:
                        bs.writegamma(d)
                        d = 1
                    else: 
                        d += 1
        return bs

class Data:
    def __init__(self, dataset, mode, path, count):
        self.dataset = dataset
        self.mode = mode
        self.count = count
        self.path = path
        self.fh = None

        # Position for sampling
        self.sampling_position = 0
        self.sampling_count = 0

    def __enter__(self):
        self.fh = open(self.path, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fh.close()

    def __getitem__(self, pos):
        self.fh.seek(0)
        bs = BitStream(self.fh)
        for i in range(pos+1):
            matrix, moves = BinaryRepresentation.read(bs, self.dataset.numplanes, self.dataset.boardsize)
        return matrix, moves

    def readall(self):
        """Returns a couple (feature matrix, labels)"""
        self.fh.seek(0)
        bs = BitStream(self.fh)
        matrix = np.ndarray((self.count, self.dataset.numplanes, self.dataset.boardsize, self.dataset.boardsize))
        labels = np.ndarray((self.count), dtype=np.int64)
        for i in range(self.count):
            matrix[i], move = BinaryRepresentation.read(bs, 7, 19)
            labels[i] = move[0] * self.dataset.boardsize + move[1]
        return matrix, labels

    def sample(self, count):
        boards = np.ndarray((count, self.dataset.numplanes, self.dataset.boardsize, self.dataset.boardsize))
        labels = np.ndarray((count), dtype=np.int64)
        if count == 0:
            return boards, move

        self.fh.seek(self.sampling_position)
        logging.debug("Reading %d samples from position %d", count, self.sampling_count)
        bs = BitStream(self.fh)

        for i in range(count):
            if self.sampling_count == self.count:
                logging.debug("Restart sampling")
                self.sampling_count = 0
                self.sampling_position = 0
                self.fh.seek(0)

            boards[i], move = BinaryRepresentation.read(bs, self.dataset.numplanes, self.dataset.boardsize)
            labels[i] = self.dataset.boardsize * move[0] + move[1]
            self.sampling_position = self.fh.tell()
            self.sampling_count += 1

        return boards, labels

    def batches(self, count):
        """Produces a stream of batches of a given size"""
        q = queue.Queue(maxsize=3)
        running = True

        if self.count < count:
            logging.warn("Batch size is too high : lowering from %d to %d", count, self.count)
            count = self.count

        def worker():
            try:
                while running:
                    # Sample
                    batch = self.sample(count)
                    q.put(batch)
            except:
                logging.error("Error in worker")
                traceback.print_exc()
            finally:     
                logging.info("Terminating queue")       
                q.put(None)

        t = threading.Thread(target=worker)
        t.start()
        
        try:
            while True:
                v = q.get()
                if not v:
                    break
                yield v
            running = False
        finally:
            if running:
                running = False
                while q.get() is not None:
                    pass

class Dataset:
    def __init__(self, path: Path):
        self.path = path
        with open(self.path.joinpath("information.json"), "rt") as fp:
            for key, value in json.load(fp).items():
                setattr(self, key, value)

        self.processor = GoBaseProcessor.load(self.processor)    

    def __getitem__(self, mode):
        return Data(self, mode, self.path.joinpath("%s.dat" % mode), getattr(self, mode))
    
