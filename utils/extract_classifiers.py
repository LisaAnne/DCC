import numpy as np
import sys
import os
import h5py
sys.path.append('utils/')
sys.path.append('utils/tools/')
import caffe_utils
from config import *
sys.path.insert(0, pycaffe_dir)
import caffe
import pdb

coco_template = '%s/%%s2014/COCO_%%s2014_%%012d.jpg' %coco_images_root

class VisualFeatureExtractor(object):

  def __init__(self, model, model_weights, device, feature_extract='probs'):
    caffe.set_mode_gpu()
    caffe.set_device(device)
    self.net = caffe.Net(model, model_weights, caffe.TEST)
    self.feature_size = self.net.blobs[feature_extract].data.shape[1]
    self.feature_extract = feature_extract
 
  def build_image_processor(self, image_dim=224):
    self.image_dim = image_dim
    self.transformer = caffe_utils.build_transformer(image_dim)
    self.image_processor = lambda x: caffe_utils.image_processor(self.transformer, x, True)

  def extract_batch_features(self, images):
    net = self.net
    data = []
    for im in images:
      data.extend(self.image_processor(im))
    net.blobs['data'].reshape(len(data),3,self.image_dim,self.image_dim)
    net.blobs['data'].data[...] = data
    out = net.forward()
    features_tmp = net.blobs[self.feature_extract].data
    features_av = np.array([np.mean(features_tmp[i:i+10], axis=0) for i in range(0, len(data), 10)])
    return features_av

def extract_features(model, model_weights, imagenet_ims=None, device=0, image_dim=224, feature_extract='probs', batch_size=10):

  net_type = 'vgg'

  im_ids_train = open('utils/image_list/coco2014_cocoid.train.txt').readlines()
  im_ids_train = [int(im_id.strip()) for im_id in im_ids_train]
  im_ids_val = open('utils/image_list/coco2014_cocoid.val_val.txt').readlines()
  im_ids_val = [int(im_id.strip()) for im_id in im_ids_val]
  im_ids_test = open('utils/image_list/coco2014_cocoid.val_test.txt').readlines()
  im_ids_test = [int(im_id.strip()) for im_id in im_ids_test]

  train_ims = [coco_template %('train', 'train', im_id) for im_id in im_ids_train]
  val_ims = [coco_template %('val', 'val', im_id) for im_id in im_ids_val]
  test_ims = [coco_template %('val', 'val', im_id) for im_id in im_ids_test]

  sets = [train_ims, val_ims, test_ims]
  set_names = ['train', 'val_val', 'val_test']
  if imagenet_ims:
    sets = []
    set_names = []
    imagenet_ims = open(imagenet_ims).readlines()
    test_imagenet = [imagenet_images_root + i.strip() for i in imagenet_ims]
    sets.append(test_imagenet)
    set_names.append('imagenet_test_ims')

  save_h5 = model_weights.split('/')[-1]

  #set up feature extractor
  extractor = VisualFeatureExtractor(model, model_weights, device, feature_extract)
  extractor.build_image_processor()
  feature_size = extractor.net.blobs['probs'].data.shape[1]

  for s, set_name in zip(sets, set_names):
    all_ims = s
    features = np.zeros((len(all_ims), feature_size))
    for ix in range(0, len(all_ims), batch_size):
      sys.stdout.write('\rOn set %s. On image %d/%d.' %(set_name, ix, len(all_ims)))
      sys.stdout.flush()

      batch_end = min(ix+batch_size, len(all_ims))
      batch_frames = all_ims[ix:batch_end]
      features_av = extractor.extract_batch_features(batch_frames)
      features[ix:ix+features_av.shape[0],:] = features_av

    h5_file = '%s/%s_feats.%s.%s.h5' %(lexical_features_root, net_type, save_h5, set_name)
    f = h5py.File(h5_file, "w")
    print "Printing to %s\n" %h5_file
    all_ims_short = [i.split('/')[-2] + '/' + i.split('/')[-1] for i in all_ims]
    assert len(all_ims_short) == len(features)
    dset = f.create_dataset("ims", data=all_ims_short)
    dset = f.create_dataset("features", data=features)
    f.close()
 

