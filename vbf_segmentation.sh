#!/bin/bash

INPUT_PATH=./test_data/01/raw.nii.gz
OUTPUT_PATH=./test_data/01/pred_post.nii.gz

while [[ $# -gt 0 ]]
do
  key="$1"

  case $key in

    -i)
    INPUT_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    -o)
    OUTPUT_PATH="$2"
    shift # past argument
    shift # past value
    ;;
     *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

echo $INPUT_PATH
sub_dir=$(dirname $INPUT_PATH)
mkdir -p $sub_dir/tmp

mirtk resample-image $INPUT_PATH $sub_dir/tmp/raw_0000.nii.gz -size 1 1 3

python vbf_correctHeader.py -i $sub_dir/tmp/raw_0000.nii.gz -r $INPUT_PATH -o $sub_dir/tmp/raw_0000.nii.gz

nnUNet_predict -i $sub_dir/tmp -o $sub_dir/tmp -t 501 -m 3d_fullres --disable_tta -f 0

python vbf_postProcessing.py -label $sub_dir/tmp/raw.nii.gz -out $OUTPUT_PATH

mirtk transform-image $OUTPUT_PATH $OUTPUT_PATH -target $INPUT_PATH -interpolation NN

rm -r $sub_dir/tmp
