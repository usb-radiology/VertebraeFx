#!/bin/bash


input_dir=./test_data

for i in $input_dir/* ; do
  if [ -d "$i" ]; then
    echo "$i"

    source /software/venv/nnunet/bin/activate
    bash ./vbf_segmentation.sh -i $i/raw.nii.gz -o $i/pred_post.nii.gz

    /software/vertebrae/venv/bin/python classify.py -m ./model/epoch=03-valid_loss=0.1720.ckpt -i $i/raw.nii.gz -s $i/pred_post.nii.gz -o $i

  fi
done
