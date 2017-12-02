# GammaGo 

## Information

This is fork of [betago](http://betago.github.com). The purpose is to simplify 
the code (pre-processing) and to allow students to experiment with
developing a full learning go bot.

## Prerequisites

The only pre-requisite needed is numpy; you should install your favorite 
machine learning toolkit. At the moment, there is a base class and examples 
for PyTorch (`linear`).

Another option is to use docker; an image named `bpiwowar/pytorch`, with `PyTorch` pre-installed,
can be used.

```
docker run -p 8080:8080  -v DIR:/wordir bpiwowar/pytorch ./ggo ARGUMENTS
```

where `DIR` is the main gammago directory.


## Models 

Models must be located in the `gammago/models` directory. For instance, the
model `linear` corresponds to importing using the class `Model` in
the `gammago.models.linear` module.


# Commands

## Downloading KGS data

To download or update GO games

```
./ggo kgs
```

## Pre-processing

Before training, you need to pre-process the data with the command 

```./ggo prepare [options]```

where the `options` are

- `--processor PROCESSOR` to select a processor  (default SevenPlane)
- `--cores CORES` to use CORES threads (default to the number of processors)
- `--boardsize SIZE` to set the board size of processed games (19 by default)
- `--buffer-size SIZE` to set the size of the board buffer (default 25000) used to randomize the boards in the sets:
  each sampled board is first put in the buffer; when the buffer is full, one board is sampled from the buffer and output.

To control the number of sampled boards, you can use either `--train`, `--validation`
or `--test` with the following argument `GAME_RATIO[:MAX_GAMES[:BOARD_RATIO[:MAX_BOARDS]]]` where

- `GAME_RATIO` is the ratio of sampled games
- `MAX_GAMES` is the maximum number of sampled games
- `GAME_RATIO` is the ratio of sampled boards (within a game)
- `MAX_GAMES` is the maximum number of sampled boards

Each argument is optional; for instance `--train .5::.3` will sample 50% of the games, and among those games, 30% of the boards

## Training

### Direct training

```./ggo direct-policy-train [--batchsize BATCHSIZE] [--checkpoint CHECKPOINT] MODEL MODEL_FILE [MODEL OPTIONS]```

where

- `--batchsize BATCHSIZE` gives the training batchsize
- `--checkpoint CHECKPOINT` gives the number of iterations before 
- `--iterations ITERATIONS` is the number of iterations
- `--reset` resets the model instead of continuing to train with it
- `MODEL_PATH` is the file or directory that will contain all the information necessary to run the model
- `MODEL OPTIONS` are options specific to the model (must be serialized with the information)

### Training by competition

```./ggo direct-policy-train [--batchsize BATCHSIZE] [--checkpoint CHECKPOINT] PARAMETERS```

where

- `--batchsize BATCHSIZE` gives the training batchsize
- `--checkpoint CHECKPOINT` gives the number of iterations before saving the model
- `--iterations ITERATIONS` is the number of iterations
- `--reset` resets the model instead of continuing to train with it
- `PARAMETER` is the file (or directory, depending on the model) that will contain all the information necessary to run the model (demo, etc.)
- `MODEL OPTIONS` are options specific to the model (must be serialized with the information )


## Demo

Launch a go game with a demo bot (you have to open a web page)

```./ggo demo [--port] MODEL [PARAMETERS]```

where `MODEL` is a model name (e.g. `gammago.models.idiot`) and `PARAMETERS` 
are the model parameters

## Simulate

Simulate a game between two bots

```./ggo demo [--port] BLACK_MODEL BLACK_PARAMETERS WHITE_MODEL WHITE_PARAMETERS```

where `MODEL` is a model name (e.g. `gammago.models.idiot`) and `PARAMETERS` 
are the model parameters (file or directory) or `""` if no parameters are
needed for the model.


# Misc

## Pre-defined models

- `idiot` is a random bot
- `gnugo` is a bot that uses gnu go
- `pachi` is a bot that uses pachi
- `linear` is a simple multi-layer neural network (one option: number of hidden layers)

## Literature

[1] A. Clark, A. Storkey [Teaching Deep Convolutional Neural Networks to Play Go](http://arxiv.org/pdf/1412.3409v2.pdf).

[2] C.J. Maddison, A. Huang, I. Sutskever, D. Silver [Move Evaluation in Go using Deep Neural Networks](http://arxiv.org/pdf/1412.6564v2.pdf)

[3] D. Silver, A. Huang, C.J. Maddison,	A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe,	J. Nham, N. Kalchbrenner, I. Sutskever,	T. Lillicrap, M. Leach,	K. Kavukcuoglu,	T. Graepel	& D. Hassabis [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html)
