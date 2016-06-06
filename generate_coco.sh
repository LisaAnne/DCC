#!/usr/bin/env bash

#coco
deploy_image=dcc_vgg.deploy.prototxt
deploy_words=dcc_vgg.wtd.prototxt
#for delta model
#deoploy_words = dcc_vgg.delta.wtd.prototxt
model_name=dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel
vocab=vocabulary.txt
precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.val_val.h5
precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.val_test.h5
precomputed_feats=vgg_feats.attributes_JJ155_NN511_VB100_coco_715_baseline_0223_iter_80000.caffemodel.val_test.h5
precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel.val_test.h5
precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel.val_val.h5
precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.val_test.h5
image_list=coco2014_cocoid.val_val.txt
split=val_val

#For models trained with out of domain text (vocabulary is larger than coco vocab)
#deploy_image=dcc_vgg.80k.deploy.prototxt
#deploy_words=dcc_vgg.80k.wtd.prototxt
#model_name=dcc_oodLM_rm1_vgg.surf.solver_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel
#vocab=yt_coco_surface_80k_vocab.txt
#precomputed_feats=vgg_feats.attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.val_val.h5
#image_list=coco2014_cocoid.val_val.txt
#split=val_val

echo $deploy_image
echo $deploy_words
echo $model_name
echo $vocab
echo $precomputed_feats
echo $image_list

python dcc.py --image_model $deploy_image \
              --language_model $deploy_words \
              --model_weights $model_name \
              --vocab $vocab \
              --precomputed_features $precomputed_feats \
              --image_list $image_list \
              --split $split \
              --generate_coco
