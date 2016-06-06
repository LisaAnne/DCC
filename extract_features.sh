#!/bin/bash

#coco
model='prototxt/train_classifiers_deploy_417.prototxt'
model_weights='classification_models/attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel'
image_dim=224

python dcc.py --model $model \
              --model_weights $model_weights \
              --batch_size 64 \
              --image_dim $image_dim \
              --extract_features

#imagenet
model='prototxts/train_classifiers_deploy.imagenet.prototxt'
model_weights='classification_models/vgg_multilabel_FT_iter_100000.caffemodel'
imagenet_images='utils/imageList/imagenet_test_images.txt'
image_dim=224

python dcc.py --image_model $model \
              --model_weights $model_weights \
              --imagenet_images $imagenet_images \
              --batch_size 64 \
              --image_dim $image_dim \
              --extract_features
