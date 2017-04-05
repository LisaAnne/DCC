#main funciton for DCC; use to extract features and evaluate results
import sys
from utils import extract_classifiers 
from eval.captioner import * 
from eval.coco_eval import *
#from eval import eval_sentences
from dcc_transfer import transfer_weights
import argparse
import pdb 
from utils.config import *
import h5py

def extract_features(args):


  extract_classifiers.extract_features(args.image_model, args.model_weights, args.imagenet_images, args.device, args.image_dim, args.lexical_feature, args.batch_size)

def transfer(args):
  
  transfer_net = transfer_weights.transfer_net(args.language_model, args.model_weights, args.orig_attributes, args.all_attributes, args.vocab)
  eval('transfer_net.' + args.transfer_type)(args.words, args.classifiers, args.closeness_metric, args.log, num_transfer=args.num_transfer, orig_net_weights=args.orig_model) 

def generate_coco(args):
  #args.model_weights, args.image_model, args.language_model, args.vocab, args.image_list, args.precomputed_features

  language_model = models_folder + args.language_model
  model_weights = weights_folder + args.model_weights
  vocab = vocab_root + args.vocab

  image_list = open_txt(image_list_root + args.image_list)

  captioner = Captioner(language_model, model_weights, 
                        sentence_generation_cont_in='cont_sentence',
                        sentence_generation_sent_in='input_sentence',
                        sentence_generation_feature_in=['image_features'],
                        sentence_generation_out='predict',
                        vocab_file=vocab,
                        prev_word_restriction=True)

  if args.precomputed_features:
    precomputed_feats = lexical_features_root + args.precomputed_features
    features = h5py.File(precomputed_feats)
    descriptor_dict = {}
    for feature, im in zip(features['features'], features['ims']):
      descriptor_dict[im] = np.array(feature)
    features.close()
  else:
    #TODO add in code to compute features if not precomputes 
    raise Exception("You must precompute features!")

  assert len(image_list) == len(descriptor_dict.keys())

  final_captions = captioner.caption_images(descriptor_dict, descriptor_dict.keys(), batch_size=1000)
  save_caps = 'results/generated_sentences/%s_%s.json' %(args.model_weights, args.image_list)
  save_json_coco_format(final_captions, save_caps)

  gt_json = coco_annotations + 'captions_%s2014.json' %args.split
  gt_template_novel = coco_annotations + 'captions_split_set_%s_%s_novel2014.json'
  gt_template_train = coco_annotations + 'captions_split_set_%s_%s_train2014.json'

  print "Scores over entire dataset..."
  score_generation(gt_json, save_caps)

  print "Scores over word splits..."
  new_words = ['bus', 'bottle', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']
  score_dcc(gt_template_novel, gt_template_train, save_caps, new_words, args.split)

def generate_imagenet(args):
  #args.model_weights, args.image_model, args.language_model, args.vocab, args.image_list, args.precomputed_features

  model_weights = weights_folder + args.model_weights
  language_model = models_folder + args.language_model
  vocab = vocab_root + args.vocab
  precomputed_feats = lexical_features_root + args.precomputed_features

  image_list = open_txt(image_list_root + args.image_list)

  if args.precomputed_features:
    precomputed_feats = lexical_features_root + args.precomputed_features
    features = h5py.File(precomputed_feats)
    descriptor_dict = {}
    for feature, im in zip(features['features'], features['ims']):
      descriptor_dict[im] = np.array(feature)
    features.close()
  else:
    #TODO add in code to compute features if not precomputes 
    raise Exception("You must precompute features!")

  captioner = Captioner(language_model, model_weights, 
                        sentence_generation_cont_in='cont_sentence',
                        sentence_generation_sent_in='input_sentence',
                        sentence_generation_feature_in=['image_features'],
                        sentence_generation_out='predict',
                        vocab_file=vocab,
                        prev_word_restriction=True)

  final_captions = captioner.caption_images(descriptor_dict, descriptor_dict.keys(), batch_size=1000)
  save_caps = 'results/generated_sentences/%s_%s.json' %(args.model_weights, args.image_list)
  save_json_other_format(final_captions, save_caps)

def eval_imagenet(args):
  result = eval_sentences.make_imagenet_result_dict(generated_sentences + args.caps) 
  eval_sentences.find_successful_classes(result)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_model",type=str)
  parser.add_argument("--language_model",type=str)
  parser.add_argument("--model_weights",type=str)
  parser.add_argument("--image_list", type=str)
  parser.add_argument("--imagenet_images",type=str, default=None) #extract_features
  parser.add_argument("--lexical_feature",type=str, default='probs') #name of layer to extract
  parser.add_argument("--orig_attributes",type=str, default='')
  parser.add_argument("--all_attributes",type=str, default='')
  parser.add_argument("--vocab", type=str, default='')
  parser.add_argument("--words", type=str, default='')
  parser.add_argument("--precomputed_features", type=str, default=None) #list of classifiers
  parser.add_argument("--classifiers", type=str, default='') #list of classifiers
  parser.add_argument("--closeness_metric", type=str, default='closeness_embedding')
  parser.add_argument("--transfer_type", type=str, default='direct_transfer')
  parser.add_argument("--split", type=str, default='val_val')
  parser.add_argument("--caps", type=str, default='')

  parser.add_argument("--orig_model", type=str, default='')
  parser.add_argument("--new_model", type=str, default='')
  parser.add_argument("--language_feature", type=str, default='predict')
  parser.add_argument("--image_feature", type=str, default='data')

  parser.add_argument("--device",type=int, default=0)
  parser.add_argument("--image_dim",type=int, default=227)
  parser.add_argument("--batch_size",type=int, default=10)
  parser.add_argument("--num_transfer",type=int, default=1)

  parser.add_argument('--extract_features', dest='extract_features', action='store_true')
  parser.set_defaults(extract_features=False)
  parser.add_argument('--generate_coco', dest='generate_coco', action='store_true')
  parser.set_defaults(generate_coco=False)
  parser.add_argument('--generate_imagenet', dest='generate_imagenet', action='store_true')
  parser.set_defaults(generate_imagenet=False)
  parser.add_argument('--eval_imagenet', dest='eval_imagenet', action='store_true')
  parser.set_defaults(eval_imagenet=False)
  parser.add_argument('--transfer', dest='transfer', action='store_true')
  parser.set_defaults(transfer=False)
  parser.add_argument('--log', dest='log', action='store_true')
  parser.set_defaults(log=False)

  args = parser.parse_args()
  
  if args.extract_features:
    extract_features(args) 

  if args.transfer:
    transfer(args)

  if args.generate_coco:
    generate_coco(args)

  if args.generate_imagenet:
    generate_imagenet(args)
  
  if args.eval_imagenet:
    eval_imagenet(args)
