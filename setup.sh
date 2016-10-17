# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
zip_file="release_DCC_06182016.zip"
unzip_file="release_DCC"
annotation_dl=1
annotation_folder="annotations"
image_dl=1
image_folder="coco_images"
tool_dl=1

function show_help {
  echo "-z: option which indicates path for downloaded zip file is.  Default is $zip_file."
  echo "-a: option which indicates whether or not to download MSCOCO annotations.  If MSCOCO annotations already downloaded, need path to annotations to properly setup folders.  Default path is $annotation-folder"
  echo "-i: option to indicate whether or not to download MSOCO images."
  echo "-t: option to indicate whether or not to download MSOCO tools."
}

while getopts "h?z:a:it" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    z)  zip_file=$OPTARG
        ;;
    a)  annotation_dl=0
        annotation_folder=$OPTARG
        ;;
    i)  image_dl=0
        ;;
    t)  tool_dl=0
        ;;
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

if [ $annotation_dl -eq 1 ]
  then
    echo "Downloading MSCOCO annotations to $annotation_folder"
    mscoco_annotation_file="annotations-1-0-3/captions_train-val2014.zip"
    wget http://msvocds.blob.core.windows.net/$mscoco_annotation_file
    unzip captions_train-val2014.zip 
  else
    echo "Not downloading MSCOCO annotations.  Annotations already in $annotation_folder"
fi

if [ $image_dl -eq 1 ]
  then
    echo "Downloading MSCOCO images to $image_folder"
    mscoco_train_image_file="coco2014/train2014.zip"
    wget http://msvocds.blob.core.windows.net/$mscoco_train_image_file
    unzip train2014.zip 
    mscoco_train_image_file="coco2014/val2014.zip"
    wget http://msvocds.blob.core.windows.net/$mscoco_train_image_file
    unzip val2014.zip
    mkdir -p $image_folder
    mv train2014 $image_folder
    mv val2014 $image_folder 
  else
    echo "Not downloading MSCOCO images.  Images already in $image_folder"
fi

if [ $tool_dl -eq 1 ]
  then
    echo "Downloading MSCOCO eval tools"
    ./utils/download_tools.sh
  else
    echo "Not downloading MSCOCO eval_tools."
fi

unzip $zip_file 

mkdir -p snapshots
mkdir -p results
mkdir -p results/generated_sentences
mv $unzip_file/trained_models .
mv $unzip_file/utils/image_list/* utils/image_list
mv $unzip_file/utils/vectors-cbow-bnc+ukwac+wikipedia.bin train_captions
mv $unzip_file/annotations_DCC/* $annotation_folder

mkdir -p outfiles
mkdir -p outfiles/transfer

#rm -r $unzip_file 
#rm $zip_file
