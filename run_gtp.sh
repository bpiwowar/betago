#!/bin/sh

path="$(dirname "$0")"
PYTHONPATH=$path
python3 -m gammago gtp "$@"

# gammago.models.basic parameters.dat
