#!/usr/bin/env bash

export PYTHONPATH='utils/:$PYTHONPATH'

caffe/python/train.py --solver prototxts/dcc_coco_rm1_vgg.solver.freezeLM.prototxt --weights snapshots/mrnn.direct_iter_110000.caffemodel --gpu 0

caffe/python/train.py --solver prototxts/dcc_coco_rm1_vgg.solver.deltaLM.prototxt --weights snapshots/dcc_coco_rm1_vgg.delta_freezeLM_iter_50000.caffemodel --gpu 0

