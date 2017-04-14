# Deep Compositional Captioning

Hendricks, Lisa Anne, et al. "Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data." CVPR (2016).

[Find the paper here.](https://arxiv.org/abs/1511.05284)

```
@inproceedings{hendricks16cvpr, 
        title = {Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data}, 
        author = {Hendricks, Lisa Anne and Venugopalan, Subhashini and Rohrbach, Marcus and Mooney, Raymond, and Saenko Kate, and Darrell, Trevor}, 
       booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
       year = {2016} 
}
```

License: BSD 2-Clause license

You should be able to replicate my results using this code.  I am still actively adding to it, so if something is unclear email me.

## Setting Up

To use my code, please make sure you have the following: 

1. Lisa Anne Hendricks' branch of Caffe installed: "https://github.com/LisaAnne/lisa-caffe-public/tree/master".  My code will probably work well with other Caffe versions, but I have tested on this version.
2. All data/models can be downloaded with setub.sh.
3. Optional -- ImageNet dataset (http://image-net.org/download).  For the ImageNet experiments, some classes are outside the 1,000 classes chosen for the ILSVRC challenge. To see which images I used, look at "utils/all_imagenet_images.txt" which includes path to imagenet image and label I used when training.

To begin, please run: ./setup.sh

My script assumes that you have already downloaded MSCOCO description annotations, images, and evaluation tools.  If not, no worries!  You can download those by using the following flags:

	--download_mscoco_annotations: downloads mscoco annotations to annotations.
	--download_mscoco_images: downloads mscoco images to images/coco_images.
	--download_mscoco_tools: downloads mscoco eval tools to utils/coco_tools.

The script will also download my annotations used for my zero-shot splits, my models, and run the transfer code so you can describe novel objects.

Next, copy "utils/config.example.py" to "utils/config.py" and make sure all paths match the paths on your machine.  In particular, you will need to indicate the path to your caffe directory, the MSCOCO dataset and evaluation toolbox (if you did not download these using setup.sh), and imagenet images.

Now that everything is setup, we can evaluate the DCC model.  

Please look at "GenerateDescriptions.ipynb" for an example.  Some details follow:

1.  The first step in DCC is to train lexical models which map images to a set of visual concepts (e.g., "sheep", "grass", "stand").
        - "attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.caffemodel": image model trained with MSCOCO images
	- "attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel": image model trained with MSCOCO images (Do not use multiple labels for held out classes.  We mine MSCOCO labels from descriptions, and therefore images can have multiple labels.  However, for the eight held out concepts, we just train with a single label corresponding to the held out class -- e.g., "bus" instead of "bus", "street", "building".  We do this to ensure that the visual model does not exploit co-occurrences)
	- "attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.caffemode": image model trained with MSCOCO images EXCEPT for objects which are held outduring paired training.  These categories are trained with ImageNet data.
	- "vgg_multilabel_FT_iter_100000.caffemodel":  image model trained on all MSCOCO images and over 600 ImageNet objects not in MSCOCO

	The code to train these models will be coming soon, but you can use all my pretrained models. 

2.  The next step in DCC is to train language models.
	- "mrnn.direct_iter_110000.caffemodel": language model trained on MSCOCO text
	- "mrnn.lm.direct_surf_lr0.01_iter_120000.caffemodel": language model trained on WebCorbus text
	- "mrnn.lm.direct_imtextyt_lr0.01_iter_120000.caffemodel": langauge model trained on Caption text

	The code to train these models will be coming soon, but you can use all my pretrained models. 
 
3.  The final training step is to train the caption model.  You can find the prototxts to train the caption models in "prototxts".  To speed up training, I pre-extract image features.  Please look at "extract_features.sh" to see how to extract features.  Train the caption models using one of the following bash scripts:
	- "run_dcc_coco_baseline_vgg.sh": model with pair supervision
	- "run_dcc_coco_rm1_vgg.sh": direct transfer model with in domain text pre-training and in domain image pre-training
	- "run_dcc_coco_rm1_vgg.delta.sh": delta transfer model with in domain text pre-training and in domain image pre-training
	- "run_dcc_imagenet_rm1_vgg.sh": direct transfer model with in domain text pre-training and out of domain image pre-training
	- "run_dcc_imagenet_rm1_vgg.im2txt.sh": direct transfer model with out of domain text pre-training with Caption txt and out of domain image pre-training
	- "run_dcc_imagenet_rm1_vgg.sh": direct transfer model with out of domain text pre-training with WebCorpus and out of domain image pre-training
        - "run_dcc_imagenet_sentences_vgg.sh": direct transfer model for describing Imagnet objects

    Note that I include all my caption models in "snapshots", so you do not have to retrain these models yourself!

4.  Novel word transfer.  Please look at transfer.sh to see how to transfer weigths for the direct transfer model and transfer_delta.sh to see how to transfer weights for the delta_transfer model.  The setup script will automatically do both direct transfer and delta transfer, so these models should be in snapshots as well.  

5.  Evaluation on MSCOCO.  Look at generate_coco.sh.

6.  Generating descriptions for ImageNet images.  Look at generate_imagenet.sh.

If you would like to compare to my descriptions, please look into the ''generated_sentences'' folder.  

Finally, if you are working on integrating novel words into captions, I suggest you also check out the following papers:

[Captioning Images with Diverse Objects](https://arxiv.org/abs/1511.05284) **Oral CVPR 2017**

[Guided Open Vocabulary Image Captioning with Constrained Beam Search](https://arxiv.org/abs/1612.00576)

If you have a paper in which you compare to DCC, let me know and I will add it to this list.


Please contact lisa_anne@berkeley.edu if you have any issues.  Happy captioning!

Updated 4/14/2016

