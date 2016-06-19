#!/usr/bin/env bash

# change to directory $DIR where this script is stored
pushd .
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
