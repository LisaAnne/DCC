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

You should be able to replicate my results using this code.  Please let me know if you have any questions.
## Setting Up

To use my code, please make sure you have the following: 

1. Lisa Anne Hendricks' branch of Caffe installed: "https://github.com/LisaAnne/lisa-caffe-public/tree/master".  My code will probably work well with other Caffe versions, but I have tested on this version.
~2. All data/models can be downloaded with setup.sh.~
2.  After I graduated my website was deleted, so please download data from a drive folder [here](https://drive.google.com/drive/u/1/folders/1ct0KhDW8ZHW4D9pxu0IX1ntTaH-XOAVV).
3. Optional -- ImageNet dataset (http://image-net.org/download).  For the ImageNet experiments, some classes are outside the 1,000 classes chosen for the ILSVRC challenge. To see which images I used, look at "utils/all_imagenet_images.txt" which includes path to imagenet image and label I used when training.

To begin, please run: ./setup.sh

My script assumes that you have already downloaded MSCOCO description annotations, images, and evaluation tools.  If not, no worries!  You can download those by using the following flags:

	--download_mscoco_annotations: downloads mscoco annotations to annotations.
	--download_mscoco_images: downloads mscoco images to images/coco_images.
	--download_mscoco_tools: downloads mscoco eval tools to utils/coco_tools.

The script will also download my annotations used for my zero-shot splits and my models (before transfer).  **Note -- to replicate my results you will need to run transfer.sh and transfer_delta.sh as described in the next few steps*.

Next, copy "utils/config.example.py" to "utils/config.py" and make sure all paths match the paths on your machine.  In particular, you will need to indicate the path to your caffe directory, the MSCOCO dataset and evaluation toolbox (if you did not download these using setup.sh), and imagenet images.

Once you have setup your paths, run "transfer.sh" and "transfer_delta.sh" to run the transfer code.  **You will not get the same results as me if you do not run the transfer code.**

Now that everything is setup, we can evaluate the DCC model.  

Please look at "GenerateDescriptions.ipynb" for an example of how to caption an image.  You do not need to retrain models, and can go directly to steps 5 and 6 if you would like to evaluate models.  Some details follow:

1.  The first step in DCC is to train lexical models which map images to a set of visual concepts (e.g., "sheep", "grass", "stand").
        - "attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.caffemodel": image model trained with MSCOCO images
	- "attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel": image model trained with MSCOCO images (Do not use multiple labels for held out classes.  We mine MSCOCO labels from descriptions, and therefore images can have multiple labels.  However, for the eight held out concepts, we just train with a single label corresponding to the held out class -- e.g., "bus" instead of "bus", "street", "building".  We do this to ensure that the visual model does not exploit co-occurrences)
	- "attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.caffemode": image model trained with MSCOCO images EXCEPT for objects which are held outduring paired training.  These categories are trained with ImageNet data.
	- "vgg_multilabel_FT_iter_100000.caffemodel":  image model trained on all MSCOCO images and over 600 ImageNet objects not in MSCOCO

	The code to train these models will be coming soon, but you can use all my pretrained models.  Use "./extract_features.sh" to extract image features for MSCOCO. 

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

If you just want to compare to my descritions, look in the "results/generated_sentences" folder. You will find:

1.  dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000.caffemodel_coco2014_cocoid.val_test.txt.json: DCC with in domain text and in domain images.
2. dcc_oodLM_rm1_vgg.surf.471.solver_0409_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel_coco2014_cocoid.val_test.txt.json: DCC with out of domain text and out of domain images
3. vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000.transfer_words_imagenet.txt_closeness_embedding.caffemodel_test_imagenet_images.txt.json: DCC images for ImageNet images.

Finally, if you are working on integrating novel words into captions, I suggest you also check out the following papers:

[Captioning Images with Diverse Objects](https://arxiv.org/abs/1511.05284) **Oral CVPR 2017**

[Incorporating Copying Mechanism in Image Captioning
for Learning Novel Objects](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yao_Incorporating_Copying_Mechanism_CVPR_2017_paper.pdf) **CVPR 2017**

[Guided Open Vocabulary Image Captioning with Constrained Beam Search](https://arxiv.org/abs/1612.00576) **EMNLP 2017**

[Neural Baby Talk](https://arxiv.org/pdf/1803.09845.pdf) **Spotlight CVPR 2018**

[Decoupled Novel Captioner](https://arxiv.org/pdf/1804.03803.pdf) **ACM MM 2018**

[Partially Supervised Image Captioning](https://arxiv.org/pdf/1806.06004.pdf) **NIPS 2018**

[Image Captioning with Unseen Objects](https://arxiv.org/pdf/1908.00047.pdf)  **Spotlight BMVC 2019**

If you have a paper in which you compare to DCC, let me know and I will add it to this list.

Please contact lisa_anne@berkeley.edu if you have any issues.  Happy captioning!

Updated 8/06/2019

