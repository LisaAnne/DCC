# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
zip_file="release_DCC_06182016.zip"
annotation_dl=1
annotation_folder="annotations"
image_dl=1
image_folder="coco_images"
tool_dl=1

function show_help {
  echo "-z: option which indicates path for downloaded zip file is.  Default is $zip_file."
  echo "-a: option which indicates whether or not to download MSCOCO annotations.  If MSCOCO annotations already downloaded, need path to annotations to properly setup folders.  Default path is $annotation-folder"
  echo "-i: option to indicate whether or not to download MSOCO images."
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
    wget http://msvocds.blob.core.windows.net/annotations-1-0-3/$mscoco_annotation_file
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
    mkdir $image_folder
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

unzip $zip_file dcc_release

mkdir snapshots
mv $current_release_zip/trained_models/caption_models/* snapshots
mv $current_release_zip/trained_models/classifiers/* snapshots
mv $current_release_zip/trained_models/language_models/* snapshots
mv $current_release_zip/utils/image_list utils
mv $current_release_zip/vectors-cbow-bnc+ukwac+wikipedia.bin train_captions
mv $current_release_zip/annotations_DCC $annotation_folder

mkdir outfiles
mkdir outfiles/transfer

rm -r dcc_release 
rm $zip_file
