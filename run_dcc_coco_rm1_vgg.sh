#!/usr/bin/env bash

export PYTHONPATH='utils/:$PYTHONPATH'

caffe/python/train.py --solver prototxts/dcc_coco_rm1_vgg.solver.prototxt --weights snapshots/mrnn.direct_iter_110000.caffemodel --gpu 2
