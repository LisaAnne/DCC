#!/bin/bash

#coco
model=prototxts/dcc_vgg.delta.wtd.prototxt
model_weights=attributes_JJ100_NN300_VB100_eightClusters_captions_cocoImages_1026_ftLM_1110_vgg_iter_5000
orig_weights=attributes_JJ100_NN300_VB100_eightClusters_imagenetImages_captions_freezeLMPretrain_vgg_iter_50000
orig_attributes='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
all_attributes='utils/lexicalList/lexicalList_parseCoco_JJ100_NN300_VB100.txt'
vocab='utils/vocabulary/vocabulary.txt'
words='utils/transfer_experiments/transfer_words_coco1.txt'
classifiers='utils/transfer_experiments/transfer_classifiers_coco1.txt'
closeness_metric='closeness_embedding'
transfer_type='delta_transfer'
num_transfer=1

python dcc.py --language_model $model \
              --model_weights $model_weights \
              --orig_attributes $orig_attributes \
              --all_attributes $all_attributes \
              --vocab $vocab \
              --words $words \
              --classifiers $classifiers \
              --transfer_type $transfer_type \
              --orig_model $orig_weights \
              --num_transfer $num_transfer \
              --transfer \
              --log



