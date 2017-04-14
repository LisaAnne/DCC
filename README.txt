Code for:

Hendricks, Lisa Anne, et al. "Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data." CVPR (2016).

License: BSD 2-Clause license

You should be able to replicate my results using this code.  I am still actively adding to it, so if something is unclear email me.

To use my code, please make sure you have the following: 

Lisa Anne Hendricks' recurrent branch of Caffe installed: "https://github.com/LisaAnne/lisa-caffe-public/tree/lisa_recurrent".  My code will probably work well with other Caffe versions, but I have tested on this version.

To begin, please copy "utils/config.example.py" to "utils/config.py" and update to reflect the paths in your machine.  For example, if you have already downloaded the MSCOCO images and do not want to redownload them, please update "coco_iamges_root" to point to where the MSCOCO images are stored on your machine.  If you do not have the MSCOCO images, we will download them in the next step.

Next, please run: ./setup.sh

This will download data needed for DCC (MSCOCO), pretrained models and other tools.  If you do not already have the MSCOCO annotations, images, and eval tools installed on your machine, you can use this script to download those tools using the following flags:
	--download_mscoco_annotations: downloads mscoco annotations
        --download_mscoco_images: downloads mscoco images
        --download_mscoco_tools: downloads mscoco eval tools



Now that everything is setup, we can evaluate the DCC model.

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

4.  Novel word transfer.  Please look at transfer.sh to see how to transfer weigths for the direct transfer model and transfer_delta.sh to see how to transfer weights for the delta_transfer model.

5.  Evaluation on MSCOCO.  Look at generate_coco.sh.

6.  Generating descriptions for ImageNet images.  Look at generate_imagenet.sh.

Please contact lisa_anne@berkeley.edu if you have any issues.  Happy captioning!

Updated 6/18/2016
