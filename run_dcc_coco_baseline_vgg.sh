#!/usr/bin/env bash

export PYTHONPATH='utils/:$PYTHONPATH'

caffe/build/tools/caffe train -solver prototxts/dcc_coco_baseline_vgg.solver.prototxt -weights trained_models/language_models/mrnn.direct_iter_110000.caffemodel -gpu 0
