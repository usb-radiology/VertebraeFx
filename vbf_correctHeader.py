import nibabel as nib
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', default='', required=True)
    parser.add_argument('-r', dest='ref', default='', required=True)
    parser.add_argument('-o', dest='output', default='', required=True)

    args = parser.parse_args()

    nb = nib.load(args.input)
    refnb = nib.load(args.ref)
    newnb = nib.Nifti1Image(np.int16(nb.get_fdata())+np.int16(refnb.dataobj.inter), nb.affine,refnb.header)

    nib.save(newnb,args.output)

