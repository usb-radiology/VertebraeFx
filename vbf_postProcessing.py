import os 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure
import skimage.measure
from skimage import morphology
import argparse


def rename(mask, hxyz):
    mask = np.squeeze(img>0)
    se = np.ones(np.ceil(2/hxyz).astype(int))
    mask=ndimage.binary_erosion(mask, se)
    mask=morphology.remove_small_objects(mask,min_size=1500/np.prod(hxyz))
    v_max = np.max(np.unique(img))
    dim=mask.shape

    if v_max<24:
        v_max=24
        
    label = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(label)
    centroid = np.asarray([x.centroid for x in regions])
    
    sorted_centroids = np.argsort(-centroid[:,1])

    en=np.unique(label)[1:]
    label_ = np.zeros(dim)

    for s in range(len(sorted_centroids)):
        label_[label==en[sorted_centroids[s]]]=v_max-s

    label_=morphology.dilation(label_, se)
    return label_

def remove_cutted_vertebrae(label, hxyz):
    longitudal_offset = int(10/hxyz[1])
    print(longitudal_offset)
    uniques = np.unique(label[:, 0:longitudal_offset, :])
    if len(uniques)>1 and np.sum(label==uniques[1])*np.prod(hxyz)/1000 < 40:
            label[label==uniques[1]]=0
    uniques = np.unique(label[:, -1:longitudal_offset, :])
    if len(uniques)>1 and np.sum(label==uniques[1])*np.prod(hxyz)/1000 < 40:
            label[label==uniques[1]]=0
    return label    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-label', metavar='label', default='', required=True)
    parser.add_argument('-out', metavar='out', default='', required=True)
    args = parser.parse_args()

    Plabel = nib.load(args.label)
    img = Plabel.get_fdata()
    hxyz = Plabel.header['pixdim'][1:4]
    renamed_label = rename(img>0, hxyz)

    finished_label = remove_cutted_vertebrae(renamed_label, hxyz).astype(int)

    P = nib.Nifti1Image(finished_label,Plabel.affine,Plabel.header)
    nib.save(P,args.out)

