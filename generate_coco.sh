#!/usr/bin/env bash

#coco

#This will generate results when using in domain data for transfer using the direct transfer method.

#These numbers are a bit better than what is reported in the paper
#model for direct transfer
deploy_words=dcc_vgg.wtd.prototxt
model_name=dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel
#model for delta transfer
#deploy_words=dcc_vgg.delta.wtd.prototxt
#model_name=dcc_coco_rm1_vgg.delta_freezeLM_iter_50000.transfer_words_coco1.txt_closeness_embedding_delta_1.caffemodel
vocab=vocabulary.txt
precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel.val_val.h5
image_list=coco2014_cocoid.val_val.txt
split=val_val

#To generate result using out of domain for transfer:

#For models trained with out of domain text (vocabulary is larger than coco vocab) you will want to use the following deploy and vocab
#deploy_words=dcc_vgg.80k.wtd.prototxt
#vocab=yt_coco_surface_80k_vocab.txt
#You will also need to use a different model to train with out of domain text.
#model_name=dcc_oodLM_rm1_vgg.im2txt.471.solver_0409_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel
#model_name=dcc_oodLM_rm1_vgg.surf.solver_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel

#To generate results when image model is trained with out of domain image data
#precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.val_val.h5


echo $deploy_words
echo $model_name
echo $vocab
echo $precomputed_feats
echo $image_list

python dcc.py --language_model $deploy_words \
              --model_weights $model_name \
              --vocab $vocab \
              --precomputed_features $precomputed_feats \
              --image_list $image_list \
              --split $split \
              --generate_coco
