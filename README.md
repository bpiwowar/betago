# GammaGo 

## Information

This is fork of [betago](http://betago.github.com). The purpose is to simplify 
the code (pre-processing) and to allow students to experiment with
developing a full learning go bot.

## Prerequisites

The only pre-requisite needed is numpy; you should install your favorite 
machine learning toolkit. At the moment, there is a base class and examples 
for PyTorch.

Using docker, you can use 

```docker ```

# Commands

## Downloading KGS data

To download or update GO games
```
./gammago kgs
```

## Pre-processing

Before training, you need to pre-process the data with the command 

```./gammago prepare [options]```

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




# Misc

## Literature

[1] A. Clark, A. Storkey [Teaching Deep Convolutional Neural Networks to Play Go](http://arxiv.org/pdf/1412.3409v2.pdf).

[2] C.J. Maddison, A. Huang, I. Sutskever, D. Silver [Move Evaluation in Go using Deep Neural Networks](http://arxiv.org/pdf/1412.6564v2.pdf)

[3] D. Silver, A. Huang, C.J. Maddison,	A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe,	J. Nham, N. Kalchbrenner, I. Sutskever,	T. Lillicrap, M. Leach,	K. Kavukcuoglu,	T. Graepel	& D. Hassabis [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html)
