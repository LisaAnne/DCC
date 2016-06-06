#!/usr/bin/env python

import pdb
import sys
from config import *
sys.path.append(pycaffe_dir)
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy
import json
import time
import re
import math

UNK_IDENTIFIER = '<unk>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  if sentence[-1] != '.':
    return sentence
  return sentence[:-1]

def tokenize_text(sentence, vocabulary, leave_out_unks=False):
 sentence = split_sentence(sentence) 
 token_sent = []
 for w in sentence:
   try:
     token_sent.append(vocabulary[w])
   except:
     if not leave_out_unks:
       try:
         token_sent.append(vocabulary['<unk>'])
       except:
         pass
     else:
       pass
 if not leave_out_unks:
   token_sent.append(vocabulary['EOS'])
 return token_sent

def open_vocab(vocab_txt):
  vocab_list = open(vocab_txt).readlines()
  vocab_list = ['EOS'] + [v.strip() for v in vocab_list]
  vocab = {}
  for iv, v in enumerate(vocab_list): vocab[v] = iv 
  return vocab

def textPreprocessor(params):
  #input: 
  #     params['caption_json']: text json which contains text and a path to an image if the text is grounded in an image
  #     params['vocabulary']: vocabulary txt to use
  #output:
  #     processed_text: tokenized text with corresponding image path (if they exist)

  #make vocabulary dict
  vocab = open_vocab(params['vocabulary'])
  json_text = read_json(params['caption_json'])
  processed_text = {}

  t = time.time()
  for annotation in json_text['annotations']:
    processed_text[annotation['id']] = {}
    processed_text[annotation['id']]['text'] = tokenize_text(annotation['caption'], vocab)
    processed_text[annotation['id']]['image'] = annotation['image_id']
  print "Setting up text dict: ", time.time()-t 
  return processed_text 

class extractData(object):

  def increment(self): 
  #uses iteration, batch_size, data_list, and num_data to extract next batch identifiers
    next_batch = [None]*self.batch_size
    if self.iteration + self.batch_size >= self.num_data:
      next_batch[:self.num_data-self.iteration] = self.data_list[self.iteration:]
      next_batch[self.num_data-self.iteration:] = self.data_list[:self.batch_size -(self.num_data-self.iteration)]
      random.shuffle(self.data_list)
      self.iteration = self.num_data - self.iteration
    else:
      next_batch = self.data_list[self.iteration:self.iteration+self.batch_size]
      self.iteration += self.batch_size
    assert self.iteration > -1
    assert len(next_batch) == self.batch_size 
    return next_batch
 
  def advanceBatch(self):
    next_batch = self.increment()
    self.get_data(next_batch)

class extractFeatureText(extractData):

  def __init__(self, dataset, params, result):
    self.extractType = 'text'
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    print 'For extractor extractText, length of data is: ', self.num_data
    self.dataset = dataset
    self.iteration = 0
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']

    #create h5 dataset
    extracted_features = h5py.File(params['feature_file'],'r')

    self.features = {}
    t = time.time()
    for ix, im in enumerate(extracted_features['ims']):
      im_key = int(im.split('_')[-1].split('.jpg')[0])
      self.features[im_key] = extracted_features['features'][ix]
    print "Setting up image dict: ", time.time()-t

    #prep to process image
    self.feature_dim = self.features.values()[0].shape[0]
    feature_data_shape = (self.batch_size, self.feature_dim)

    #preperation to output top
    self.text_data_key = params['text_data_key']
    self.text_label_key = params['text_label_key']
    self.marker_key = params['text_marker_key']
    self.feature_data_key = params['feature_data_key']
    self.top_keys = [self.text_data_key, self.text_label_key, self.marker_key, self.feature_data_key]
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']
    self.top_shapes = [(self.stream_size, self.batch_size), (self.stream_size, self.batch_size), (self.stream_size, self.batch_size), feature_data_shape]
    self.result = result 
    
  def get_data(self, next_batch):
    batch_images = [self.dataset[nb]['image'] for nb in next_batch]
    next_batch_input_sentences = np.zeros((self.stream_size, self.batch_size))
    next_batch_target_sentences = np.ones((self.stream_size, self.batch_size))*-1
    next_batch_feature_data = np.ones((self.batch_size, self.feature_dim))
    next_batch_markers = np.ones((self.stream_size, self.batch_size))
    next_batch_markers[0,:] = 0
    for ni, nb in enumerate(next_batch):
      ns = self.dataset[nb]['text']
      nf = self.dataset[nb]['image']
      num_words = len(ns)
      ns_input = ns[:min(num_words, self.stream_size-1)]
      ns_target = ns[:min(num_words, self.stream_size)]
      next_batch_input_sentences[1:min(num_words+1, self.stream_size), ni] = ns_input 
      next_batch_target_sentences[:min(num_words, self.stream_size), ni] = ns_target
      next_batch_feature_data[ni,...] = self.features[nf] 

    self.result[self.text_data_key] = next_batch_input_sentences
    self.result[self.text_label_key] = next_batch_target_sentences
    self.result[self.marker_key] = next_batch_markers
    self.result[self.feature_data_key] = next_batch_feature_data

class extractMulti(extractData):

  def __init__(self, dataset, params, result):
    #just need to set up parameters for "increment"
    self.extractors = params['extractors']
    self.batch_size = params['batch_size']
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0
    self.batch_size = params['batch_size']
  
    self.top_keys = []
    self.top_shapes = []
    for e in self.extractors:
      self.top_keys.extend(e.top_keys)
      self.top_shapes.extend(e.top_shapes)

  def get_data(self, next_batch):
    t = time.time()
    for e in self.extractors:
      e.get_data(next_batch)

class batchAdvancer(object):
  
  def __init__(self, extractors):
    self.extractors = extractors

  def __call__(self):
    #The batch advancer just calls each extractor
    for e in self.extractors:
      e.advanceBatch() 

class python_data_layer(caffe.Layer):
  
  def setup(self, bottom, top):
    random.seed(10)
  
    self.params = eval(self.param_str)
    params = self.params

    #set up prefetching
    self.thread_result = {}
    self.thread = None

    self.setup_extractors()
 
    self.batch_advancer = batchAdvancer(self.data_extractors) 
    self.top_names = []
    self.top_shapes = []
    for de in self.data_extractors:
      self.top_names.extend(de.top_keys)
      self.top_shapes.extend(de.top_shapes)
 
    self.dispatch_worker()

    if 'top_names' in params.keys():
      #check top names equal to each other...
      if not (set(params['top_names']) == set(self.top_names)):
        raise Exception("Input 'top names' not the same as determined top names.")
      else:
        self.top_names == params['top_names']

    print self.top_names
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    #for top_index, name in enumerate(self.top_names.keys()):

    for top_index, name in enumerate(self.top_names):
      shape = self.top_shapes[top_index] 
      print 'Top name %s has shape %s.' %(name, shape)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    for top_index, name in zip(range(len(top)), self.top_names):
      top[top_index].data[...] = self.thread_result[name] 

    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

class pairedCaptionData(python_data_layer):
  
  def setup_extractors(self):

    params = self.params

    #check that all parameters are included and set default params
    assert 'caption_json' in self.params.keys()
    assert 'vocabulary' in self.params.keys()
    assert 'feature_file' in self.params.keys()
    if 'batch_size' not in params.keys(): params['batch_size'] = 100 
    if 'stream_size' not in params.keys(): params['stream_size'] = 20 

    params['text_data_key'] = 'input_sentence'
    params['text_label_key'] = 'target_sentence'
    params['text_marker_key'] = 'cont_sentence'
    params['feature_data_key'] = 'data'
    
    data = textPreprocessor(params)
    data_extractor = extractFeatureText(data, params, self.thread_result)

    self.data_extractors = [data_extractor]
