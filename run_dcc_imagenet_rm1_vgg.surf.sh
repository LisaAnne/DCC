#!/usr/bin/env bash

export PYTHONPATH='utils/:$PYTHONPATH'

caffe/python/train.py --solver prototxts/dcc_oodLM_rm1_vgg.surf.solver.prototxt --weights snapshots/mrnn.lm.direct_surf_lr0.01_iter_120000.caffemodel --gpu 0
