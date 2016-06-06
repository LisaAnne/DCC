#!/usr/bin/env bash

deploy_image=mrnn_attributes_fc8.direct.from_features.80k.deploy.prototxt
deploy_words=mrnn_attributes_fc8.direct.from_features.80k.wtd.prototxt
#model_name=alex_feats.alex_multilabel_FT_iter_50000_imagenetSentences_iter_110000.caffemodel
#model_name=alex_feats.alex_multilabel_FT_iter_50000_imagenetSentences_iter_110000.transfer_words_imagenet.txt_closeness_embedding.caffemodel
model_name=vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000.caffemodel
#model_name=vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000.transfer_words_imagenet.txt_closeness_embedding.caffemodel
vocab=yt_coco_surface_80k_vocab.txt
#precomputed_feats=alex_feats.alex_multilabel_FT_iter_50000.caffemodel.test_imagenet_ims.h5
precomputed_feats=vgg_feats.vgg_multilabel_FT_iter_100000.caffemodel.imagenet_ims_test.h5
image_list=test_images_rebuttal.txt
split=val_val
language_feature='predict'

python dcc.py --image_model $deploy_image \
              --language_model $deploy_words \
              --model_weights $model_name \
              --vocab $vocab \
              --precomputed_features $precomputed_feats \
              --image_list $image_list \
              --language_feature $language_feature \
              --generate_imagenet
