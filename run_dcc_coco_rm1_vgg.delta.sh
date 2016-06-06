#!/usr/bin/env bash

export PYTHONPATH='utils/:$PYTHONPATH'

caffe/build/tools/caffe train -solver prototxts/dcc_coco_rm1_vgg.solver.freezeLM.prototxt -weights trained_models/language_models/mrnn.direct_iter_110000.caffemodel -gpu 0

caffe/build/tools/caffe train -solver prototxts/dcc_coco_rm1_vgg.solver.deltaLM.prototxt -weights snapshots/dcc_coco_rm1_vgg.freezeLM.prototxt_iter_50000.caffemodel -gpu 0

