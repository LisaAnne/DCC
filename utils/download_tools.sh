#!/usr/bin/env bash

# change to directory $DIR where this script is stored
pushd .
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make

echo "Finished downloading coco tools."

cd ..
git clone https://github.com/tylin/coco-caption.git
echo "Finished downloading caption eval tools"

