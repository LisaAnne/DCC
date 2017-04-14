import os

caffe_dir = 'caffe/'  #path to your caffe directory
pycaffe_dir = caffe_dir + 'python/'  #path to your pycaffe directory
lexical_features_root = 'lexical_features/' #path to store extracted features.  You will need to extract these with "extract-features.sh"

#All data in "setup.sh" will be downloaded to these folders by default.  If you would like them to be downloaded somewhere else, you will need to upate the paths here and in setup.sh!
coco_annotations = 'annotations/'  #path to MSCOCO annotations (will be downloaded by "setup.sh" if not already installed)
coco_images_root = 'images/coco_images/' #path to MSCOCO images (will be downloaded by "setup.sh" if not already installed)
imagenet_images_root = 'images/imagenet_images/' #subset of imagenet dataset collected for DCC.  "setup.sh" does not download these images as of now.
tools_folder = 'utils/coco_tools/'  #path to MSCOCO eval tools (will be downloaded by "setup.sh" if not already installed)
models_folder = 'prototxts/' 
weights_folder = 'snapshots/' 
vocab_root = 'utils/vocabulary/'
image_list_root = 'utils/image_list/'
os.environ["COCO_EVAL_PATH"] = tools_folder
