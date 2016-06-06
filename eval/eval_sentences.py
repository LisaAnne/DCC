from utils.config import *
from utils.python_utils import *
import sys
import pdb
import re
import numpy as np
import os

COCO_EVAL_PATH = coco_caption_eval 
sys.path.insert(0,COCO_EVAL_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

rm_word_dict = {'bus': ['bus', 'busses'],
                'bottle': ['bottle', 'bottles'],
                'couch': ['couch', 'couches', 'sofa', 'sofas'],
                'microwave': ['microwave', 'microwaves'],
                'pizza': ['pizza', 'pizzas'],
                'racket': ['racket', 'rackets', 'racquet', 'racquets'],
                'suitcase': ['luggage', 'luggages', 'suitcase', 'suitcases'],
                'zebra': ['zebra', 'zebras']} 

def split_sent(sent):
  sent = sent.lower()
  sent = re.sub('[^A-Za-z0-9\s]+','', sent)
  return sent.split()

def score_generation(gt_filename=None, generation_result=None):

  coco_dict = read_json(generation_result)
  coco = COCO(gt_filename)
  generation_coco = coco.loadRes(generation_result)
  coco_evaluator = COCOEvalCap(coco, generation_coco)
  #coco_image_ids = [self.sg.image_path_to_id[image_path]
  #                  for image_path in self.images]
  coco_image_ids = [j['image_id'] for j in coco_dict]
  coco_evaluator.params['image_id'] = coco_image_ids
  results = coco_evaluator.evaluate(return_results=True)
  return results

def F1(generated_json, novel_ids, train_ids, word):
  set_rm_words = set(rm_word_dict[word])
  gen_dict = {}
  for c in generated_json:
    gen_dict[c['image_id']] = c['caption']

  #true positive are sentences that contain match words and should
  tp = sum([1 for c in novel_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) > 0]) 
  #false positive are sentences that contain match words and should not
  fp = sum([1 for c in train_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) > 0])
  #false positive are sentences that do not contain match words and should
  fn = sum([1 for c in novel_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) == 0 ])
 
  #precision = tp/(tp+fp)
  if tp > 0:  
    precision = float(tp)/(tp+fp) 
    #recall = tp/(tp+fn)
    recall = float(tp)/(tp+fn)
    #f1 = 2* (precision*recall)/(precision+recall)
    return 2*(precision*recall)/(precision+recall)
  else:
    return 0.

def score_result_subset(result, ids, metrics=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']):

  metric_dict = {}
  for m in metrics: metric_dict[m] = []

  for id in ids:
    for m in metrics:
      metric_dict[m].append(result[id][m])
  for m in metric_dict: metric_dict[m] = np.mean(metric_dict[m])

  for m in metrics:
    print "%s: %.04f" %(m, metric_dict[m])

def make_imagenet_result_dict(imagenet_sents):
  caps = read_json(imagenet_sents)

  imagenet_result_dict = {}
  for cap in caps:
    key = cap['image_id'].split('/')[0]
    if key not in imagenet_result_dict.keys():
      imagenet_result_dict[key] = {}

    imagenet_result_dict[key][cap['image_id']] = cap['caption']

  return imagenet_result_dict

def find_successful_classes(imagenet_result_dict, num_train_examples=100):

  #build num_train_examples dict
  train_images = open_txt('utils/imageList/train_images_rebuttal.txt')
  train_count = {}
  for line in train_images:
    o = line.split('/')[0]
    if o not in train_count.keys():
      train_count[o] = 0
    train_count[o] += 1

  successful_class = 0
  total_class = 0
  for o in train_count.keys():
    if train_count[o] >= num_train_examples:
      count_caps = 0
      total_class += 1
      for im_id in imagenet_result_dict[o].keys():
        sent = split_sent(imagenet_result_dict[o][im_id])
        if o in sent:
          count_caps += 1
      if count_caps > 0:
        successful_class += 1
  print "Percent successful classes: %f" %(float(successful_class)/total_class)
  print "Correct classes: %d" %successful_class
  print "Total classes: %d" %total_class

def add_new_word(gt_filename, generation_result, words, dset_name='val_val'):
  results = score_generation(gt_filename, generation_result)
  generation_sentences = read_json(generation_result)
  for word in words:
    gt_novel_file = annotations + 'captions_split_set_%s_%s_novel2014.json' %(word, dset_name)
    gt_train_file = annotations + 'captions_split_set_%s_%s_train2014.json' %(word, dset_name)
    gt_novel_json = read_json(gt_novel_file)
    gt_train_json = read_json(gt_train_file)
   
    gt_novel_ids = [c['image_id'] for c in gt_novel_json['annotations']]
    gt_train_ids = [c['image_id'] for c in gt_train_json['annotations']]
 
    gen_novel = [] 
    gen_train = []
    for c in  generation_sentences:
      if c['image_id'] in gt_novel_ids:
        gen_novel.append(c)
      else: 
        gen_train.append(c)

    save_json(gen_novel, 'tmp_gen_novel.json')
    save_json(gen_train, 'tmp_gen_train.json')

    print "Word: %s.  Novel scores:" %word
    score_generation(gt_novel_file, 'tmp_gen_novel.json')
    print "Word: %s.  Train scores:" %word
    score_generation(gt_train_file, 'tmp_gen_train.json')
    f1 =  F1(generation_sentences, gt_novel_ids, gt_train_ids, word)
    print "Word: %s.  F1 score: %.04f\n" %(word, f1)

    os.remove('tmp_gen_novel.json')
    os.remove('tmp_gen_train.json')

