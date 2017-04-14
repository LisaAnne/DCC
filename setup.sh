#!/bin/bash
# POSIX

#This was tested on a Linux system.  You may run into issues if you try to do this on another system (e.g, MAC OS does not have "wget")

#TODO: Download imagenet images

# Initialize variables:
download_mscoco_annotations=0
download_mscoco_images=0
download_mscoco_tools=0

annotation_folder="annotations"
image_folder="images/coco_images"
tools_folder="utils/coco_tools"
models_folder="snapshots"
gen_setneces_folder="results/generated_sentences"

dcc_data=( "captions_no_caption_rm_eightCluster_train2014.json" "captions_split_set_bottle_val_test_novel2014.json" "captions_split_set_bottle_val_test_train2014.json" "captions_split_set_bottle_val_val_novel2014.json" "captions_split_set_bottle_val_val_train2014.json" "captions_split_set_bus_val_test_novel2014.json" "captions_split_set_bus_val_test_train2014.json" "captions_split_set_bus_val_val_novel2014.json" "captions_split_set_bus_val_val_train2014.json" "captions_split_set_couch_val_test_novel2014.json" "captions_split_set_couch_val_test_train2014.json" "captions_split_set_couch_val_val_novel2014.json" "captions_split_set_couch_val_val_train2014.json" "captions_split_set_microwave_val_test_novel2014.json" "captions_split_set_microwave_val_test_train2014.json" "captions_split_set_microwave_val_val_novel2014.json" "captions_split_set_microwave_val_val_train2014.json" "captions_split_set_pizza_val_test_novel2014.json" "captions_split_set_pizza_val_test_train2014.json" "captions_split_set_pizza_val_val_novel2014.json" "captions_split_set_pizza_val_val_train2014.json" "captions_split_set_racket_val_test_novel2014.json" "captions_split_set_racket_val_test_train2014.json" "captions_split_set_racket_val_val_novel2014.json" "captions_split_set_racket_val_val_train2014.json" "captions_split_set_suitcase_val_test_novel2014.json" "captions_split_set_suitcase_val_test_train2014.json" "captions_split_set_suitcase_val_val_novel2014.json" "captions_split_set_suitcase_val_val_train2014.json" "captions_split_set_zebra_val_test_novel2014.json" "captions_split_set_zebra_val_test_train2014.json" "captions_split_set_zebra_val_val_novel2014.json" "captions_split_set_zebra_val_val_train2014.json" "captions_val_test2014.json" "captions_val_val2014.json" )
dcc_models=( "caption_models/attributes_JJ100_NN300_VB100_eightClusters_captions_cocoImages_1026_ftLM_1110_vgg_iter_5000.caffemodel" "caption_models/attributes_JJ100_NN300_VB100_eightClusters_imagenetImages_captions_freezeLMPretrain_vgg_iter_50000.caffemodel" "caption_models/dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000.caffemodel" "caption_models/dcc_oodLM_rm1_vgg.im2txt.471.solver_0409_iter_110000.caffemodel" "caption_models/dcc_oodLM_rm1_vgg.surf.471.solver_0409_iter_110000.caffemodel" "caption_models/vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000.caffemodel" "classifiers/attributes_JJ100_NN300_VB100_allObjects_coco_vgg_0111_iter_80000.caffemodel" "classifiers/attributes_JJ100_NN300_VB100_clusterEight_imagenet_vgg_0112_iter_80000.caffemodel" "classifiers/attributes_JJ100_NN300_VB100_coco_471_eightCluster_0223_iter_80000.caffemodel" "classifiers/vgg_multilabel_FT_iter_100000.caffemodel" "language_models/mrnn.direct_iter_110000.caffemodel" "language_models/mrnn.lm.direct_surf_lr0.01_iter_120000.caffemodel" "language_models/mrnn.lm.direct_imtextyt_lr0.01_iter_120000.caffemodel" )
dcc_utils=( "image_list/coco2014_cocoid.train.txt" "image_list/coco2014_cocoid.val_test.txt" "image_list/coco2014_cocoid.val_val.txt" "image_list/test_imagenet_images.txt" "image_list/train_imagenet_images.txt" "vectors-cbow-bnc+ukwac+wikipedia.bin" "vocabulary/vocabulary.txt" "vocabulary/yt_coco_surface_80k_vocab.txt" )
dcc_sentences=( "dcc_coco_rm1_vgg.471.solver.prototxt_iter_110000.caffemodel_coco2014_cocoid.val_test.txt.json" "dcc_oodLM_rm1_vgg.surf.471.solver_0409_iter_110000.transfer_words_coco1.txt_closeness_embedding.caffemodel_coco2014_cocoid.val_test.txt.json" "vgg_feats.vgg_multilabel_FT_iter_100000_imagenetSentences_iter_110000.transfer_words_imagenet.txt_closeness_embedding.caffemodel_test_imagenet_images.txt.json" )

show_help () {
  echo "--download_mscoco_annotations: downloads mscoco annotations to $annotation_folder." 
  echo "--download_mscoco_images: downloads mscoco images to $image_folder."
  echo "--download_mscoco_tools: downloads mscoco eval tools to $tools_folder."
}

while :; do
  case $1 in 
    -h|-\?|--help) 
      show_help
      exit
      ;;
    --download_mscoco_annotations)
      download_mscoco_annotations=$((download_mscoco_annotations + 1))
      ;;
    --download_mscoco_images)
      download_mscoco_images=$((download_mscoco_images + 1))
      ;;
    --download_mscoco_tools)
      download_mscoco_tools=$((download_mscoco_tools + 1))
      ;;
    --)
      shift
      break
      ;;
    *)
      break
  esac
  shift
done

mkdir -p $annotation_folder
mkdir -p $image_folder
mkdir -p $tools_folder

if [ $download_mscoco_annotations -eq 1 ]
  then
    echo "Downloading MSCOCO annotations to $annotation_folder"
    mscoco_annotation_file="annotations-1-0-3/captions_train-val2014.zip"
    wget http://msvocds.blob.core.windows.net/$mscoco_annotation_file
    unzip captions_train-val2014.zip 
    mv annotations/* $annotation_folder
  else
    echo "Not downloading MSCOCO annotations."
fi

if [ $download_mscoco_images -eq 1 ]
  then
    echo "Downloading MSCOCO images to $image_folder"
    mscoco_train_image_file="coco2014/train2014.zip"
    wget http://msvocds.blob.core.windows.net/$mscoco_train_image_file
    unzip train2014.zip 
    mscoco_val_image_file="coco2014/val2014.zip"
    wget http://msvocds.blob.core.windows.net/$mscoco_val_image_file
    unzip val2014.zip
    mv train2014 $image_folder
    mv val2014 $image_folder 
  else
    echo "Not downloading MSCOCO images."
fi

if [ $download_mscoco_tools -eq 1 ]
  then
    echo "Downloading MSCOCO eval tools to $tools_folder"
    ./utils/download_tools.sh
  else
    echo "Not downloading MSCOCO eval_tools."
fi

mkdir -p $models_folder 
mkdir -p results
mkdir -p results/generated_sentences

#get data for DCC
echo "Downloading dcc data..."
cd $annotation_folder
for i in "${dcc_data[@]}"
do 
  echo "Downloading: " $i
  wget https://people.eecs.berkeley.edu/~lisa_anne/release_DCC/annotations_DCC/$i
done
cd ..

#get pretrained models for DCC
echo "Downloading dcc models..."
cd $models_folder 
for i in "${dcc_models[@]}"
do 
  echo "Downloading: " $i
  wget https://people.eecs.berkeley.edu/~lisa_anne/release_DCC/trained_models/$i
done
cd ..

#get utils for DCC
echo "Downloading dcc utils..."
cd $models_folder 
for i in "${dcc_utils[@]}"
do 
  echo "Downloading: " $i
  wget https://people.eecs.berkeley.edu/~lisa_anne/release_DCC/utils/$i
done
cd ..

mv utils/image_list/vectors-cbow-bnc+ukwac+wikipedia.bin dcc_transfer

mkdir -p results/generated_sentences 
cd results/generated_sentences

#get generated sentences 
echo "Downloading generated sentences..."
cd $models_folder 
for i in "${dcc_sentences[@]}"
do 
  echo "Downloading: " $i
  wget https://people.eecs.berkeley.edu/~lisa_anne/release_DCC/generated_sentences/$i
done
cd ../..

mkdir -p outfiles
mkdir -p outfiles/transfer

#clone utilities from other folders
git clone git@github.com:LisaAnne/sentence_gen_tools.git eval
git clone https://github.com/LisaAnne/python_tools utils/tools

ln -s utils/tools eval
