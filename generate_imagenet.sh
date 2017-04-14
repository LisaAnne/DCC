#!/usr/bin/env bash

deploy_words=dcc_vgg.80k.wtd.imagenet.prototxt
model_name=vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000.transfer_words_imagenet.txt_closeness_embedding.caffemodel
vocab=yt_coco_surface_80k_vocab.txt
precomputed_feats=vgg_feats.vgg_multilabel_FT_iter_100000.caffemodel.imagenet_ims_test.h5
#image_list=test_imagenet_images.txt
image_list=gecko_test_list.txt
language_feature='predict'

python dcc.py --language_model $deploy_words \
              --model_weights $model_name \
              --vocab $vocab \
              --precomputed_features $precomputed_feats \
              --image_list $image_list \
              --language_feature $language_feature \
              --generate_imagenet
