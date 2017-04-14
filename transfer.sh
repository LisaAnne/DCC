#!/bin/bash

#coco

#IN DOMAIN language

model=prototxts/dcc_vgg.wtd.prototxt
model_weights=dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000
orig_attributes='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
all_attributes='utils/lexicalList/lexicalList_parseCoco_JJ100_NN300_VB100.txt'
vocab='utils/vocabulary/vocabulary.txt'
words='utils/transfer_experiments/transfer_words_coco1.txt'
classifiers='utils/transfer_experiments/transfer_classifiers_coco1.txt'
closeness_metric='closeness_embedding'
transfer_type='direct_transfer'

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

#OUT OF DOMAIN language (im2txt)
model=prototxts/dcc_vgg.80k.wtd.prototxt #prototxt used when using OUT OF DOMAIN language features
model_weights=dcc_oodLM_rm1_vgg.im2txt.471.solver_0409_iter_110000 #language learned from im2txt LM
orig_attributes='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
all_attributes='utils/lexicalList/lexicalList_parseCoco_JJ100_NN300_VB100.txt'
vocab='utils/vocabulary/yt_coco_surface_80k_vocab.txt' #vocab used when training with OUT OF DOMAIN language features
words='utils/transfer_experiments/transfer_words_coco1.txt'
classifiers='utils/transfer_experiments/transfer_classifiers_coco1.txt'
closeness_metric='closeness_embedding'
transfer_type='direct_transfer'

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

#OUT OF DOMAIN language (surf)
model=prototxts/dcc_vgg.80k.wtd.prototxt #prototxt used when using OUT OF DOMAIN language features
model_weights=dcc_oodLM_rm1_vgg.surf.471.solver_0409_iter_110000 #language learned from surf LM
orig_attributes='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
all_attributes='utils/lexicalList/lexicalList_parseCoco_JJ100_NN300_VB100.txt'
vocab='utils/vocabulary/yt_coco_surface_80k_vocab.txt' #vocab used when training with OUT OF DOMAIN language features
words='utils/transfer_experiments/transfer_words_coco1.txt'
classifiers='utils/transfer_experiments/transfer_classifiers_coco1.txt'
closeness_metric='closeness_embedding'
transfer_type='direct_transfer'

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

#imagenet
model='prototxts/dcc_vgg.80k.wtd.imagenet.prototxt'
model_weights='vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000'
orig_attributes='utils/lexicalList/lexicalList_JJ100_NN300_VB100_rmEightCoco1.txt'
all_attributes='utils/lexicalList/lexicalList_471_rebuttalScale.txt'
vocab='utils/vocabulary/yt_coco_surface_80k_vocab.txt'
words='utils/transfer_experiments/transfer_words_imagenet.txt'
classifiers='utils/transfer_experiments/transfer_classifiers_imagenet.txt'
closeness_metric='closeness_embedding'
transfer_type='direct_transfer'

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



