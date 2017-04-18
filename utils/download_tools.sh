#!/usr/bin/env bash

# change to directory $DIR where this script is stored
pushd .
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

cd utils
git clone https://github.com/tylin/coco-caption.git coco_tools
cd ..
echo "Finished downloading caption eval tools"

