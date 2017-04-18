#!/usr/bin/env bash

#This will generate results when using in domain data for transfer using the direct transfer method.

#IN DOMAIN DIRECT TRANSFER
deploy_words=dcc_vgg.wtd.prototxt
model_name=dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel
>>>>>>> can train all models!
vocab=vocabulary.txt
precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel.val_test.h5

#IN DOMAIN DELTA TRANSFER
#deploy_words=dcc_vgg.delta.wtd.prototxt
#model_name=dcc_coco_rm1_vgg.delta_freezeLM_iter_50000.transfer_words_coco1.txt_closeness_embedding_delta_1.caffemodel
#vocab=vocabulary.txt
#precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel.val_test.h5


#To generate result using out of domain for transfer:

#OUT OF DOMAIN IMAGE, OUT OF DOMAIN LANGUAGE (IM2TXT)
#deploy_words=dcc_vgg.80k.wtd.prototxt
#vocab=yt_coco_surface_80k_vocab.txt
#model_name=dcc_oodLM_rm1_vgg.im2txt.471.solver_0409_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel
#precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.val_test.h5

#OUT OF DOMAIN IMAGE, OUT OF DOMAIN LANGUAGE (IM2TXT)
#deploy_words=dcc_vgg.80k.wtd.prototxt
#vocab=yt_coco_surface_80k_vocab.txt
#model_name=dcc_oodLM_rm1_vgg.surf.471.solver_0409_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel
#precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.val_test.h5

#change to "val_val" to eval on validation set
image_list=coco2014_cocoid.val_test.txt
split=val_test

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
