#!/usr/bin/env bash

export PYTHONPATH='utils/:$PYTHONPATH'

caffe/build/tools/caffe train -solver prototxts/dcc_oodLM_rm1_vgg.surf.solver.prototxt -weights trained_models/language_models/mrnn.lm.direct_surf_lr0.01_iter_120000.caffemodel -gpu 0
