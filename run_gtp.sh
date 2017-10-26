#!/bin/sh

path="$(dirname "$0")"
PYTHONPATH=$path
python3 -m betago gtp "$@"

# betago.models.basic parameters.dat
