#!/usr/bin/env bash

export PYTHONPATH='utils/:$PYTHONPATH'

caffe/python/train.py --solver prototxts/dcc_coco_baseline_vgg.solver.prototxt --weights snapshots/mrnn.direct_iter_110000.caffemodel --gpu 0
