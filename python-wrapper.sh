#!/bin/sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":/opt/conda/lib
/opt/conda/bin/python-real "$@"
