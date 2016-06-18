#!/bin/bash

#coco
model=prototxts/dcc_vgg.80k.wtd.prototxt
model_weights=dcc_oodLM_rm1_vgg.surf.471.solver_0409_iter_110000
orig_attributes='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
all_attributes='utils/lexicalList/lexicalList_parseCoco_JJ100_NN300_VB100.txt'
vocab='utils/vocabulary/yt_coco_surface_80k_vocab.txt'
words='utils/transfer_experiments/transfer_words_coco1.txt'
classifiers='utils/transfer_experiments/transfer_classifiers_coco1.txt'
closeness_metric='closeness_embedding'
transfer_type='direct_transfer'

#imagenet
#model='../captions_add_new_word/mrnn_attributes_fc8.direct.from_features.80k.wtd.prototxt'
##model_weights='alex_feats.alex_multilabel_FT_iter_50000_imagenetSentences_iter_110000'
#model_weights='vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000'
#orig_attributes='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
#all_attributes='utils/lexicalList/lexicalList_471_rebuttalScale.txt'
#vocab='utils/vocab/yt_coco_surface_80k_vocab.txt'
#words='utils/transfer_experiments/transfer_words_imagenet.txt'
#classifiers='utils/transfer_experiments/transfer_classifiers_imagenet.txt'
#closeness_metric='closeness_embedding'
#transfer_type='direct_transfer'

python dcc.py --language_model $model \
              --model_weights $model_weights \
              --orig_attributes $orig_attributes \
              --all_attributes $all_attributes \
              --vocab $vocab \
              --words $words \
              --classifiers $classifiers \
              --transfer_type $transfer_type \
              --transfer \
              --log



