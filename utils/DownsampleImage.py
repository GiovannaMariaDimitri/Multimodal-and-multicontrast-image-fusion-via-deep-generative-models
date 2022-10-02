import pandas as pd
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from nilearn.image import crop_img
from nilearn.masking import apply_mask


def _crop_img_to(niimg, slices, copy=True):
    """Crops niimg to a smaller size

    Crop niimg to size indicated by slices and adjust affine
    accordingly

    Parameters
    ==========
    niimg: niimg
        niimg to be cropped. If slices has less entries than niimg
        has dimensions, the slices will be applied to the first len(slices)
        dimensions

    slices: list of slices
        Defines the range of the crop.
        E.g. [slice(20, 200), slice(40, 150), slice(0, 100)]
        defines a 3D cube

    copy: boolean
        Specifies whether cropped data is to be copied or not.
        Default: True

    Returns
    =======
    cropped_img: niimg
        Cropped version of the input image
    """

    #niimg = check_niimg(niimg)

    data = niimg.get_data()
    affine = niimg.get_affine()

    cropped_data = data[slices]
    if copy:
        cropped_data = cropped_data.copy()

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = np.array([s.start for s in slices])
    new_origin = old_origin + linear_part.dot(new_origin_voxel)

    new_affine = np.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin

    new_niimg = nib.Nifti1Image(cropped_data, new_affine)

    return new_niimg



def crop_imgOriginal(niimg,ValueByWhichWantToDivideSize, rtol=1e-8, copy=True):
    """Crops niimg as much as possible

    Will crop niimg, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.

    Parameters
    ==========
    niimg: niimg
        niimg to be cropped.

    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.

    copy: boolean
        Specifies whether cropped data is copied or not.

    Returns
    =======
    cropped_img: niimg
        Cropped version of the input image
    """

    #niimg = check_niimg(niimg)
    data = niimg.get_data()
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    print(start[0])
    print(end[0])
    for i in range(0,len(start)):
        if (abs(end[i]-start[i])%ValueByWhichWantToDivideSize != 0):
            dim=abs(end[i]-start[i])
            NumeroInteroDivisione=dim//ValueByWhichWantToDivideSize
            NumeroInteroDivisionePiuUno=NumeroInteroDivisione+1
            DiffN=NumeroInteroDivisione*ValueByWhichWantToDivideSize-dim
            DiffNPiuUno=NumeroInteroDivisionePiuUno*ValueByWhichWantToDivideSize-dim
            if (abs(DiffN) < abs(DiffNPiuUno)):
                #print("Difference with N")
                #print(abs(DiffN))
                end[i]=end[i]-abs(dim-NumeroInteroDivisione*ValueByWhichWantToDivideSize)
                #print("Coordinata")
                #print(i)
                #print(abs(NumeroInteroDivisione*ValueByWhichWantToDivideSize)-dim)
            else:
                #print("Difference with N plus one")
                #print(abs(DiffNPiuUno))
                end[i]=end[i]+abs(dim-NumeroInteroDivisionePiuUno*ValueByWhichWantToDivideSize)
    slices = [slice(s, e) for s, e in zip(start, end)]
    print("Slices")
    print(slices)
    return _crop_img_to(niimg, slices, copy=copy)





#FUNCTION THAT DOES ALL THE JOB OF TAKING IMAGE, MASK, DOWNSAMPLING VALUE, AND THE
#VALUE BY WHICH I WANT THE ORIGINAL MATRIX TO BE DIVISIBLE
def DownSampleMultiplyResizeCrop(ImgFileWithPath,MaskFileWithPath,DownSampleValue,ValueByWhichWantToDivideSize):
    img=nib.load(ImgFileWithPath) #Load the Image
    mask=nib.load(MaskFileWithPath)
    print(img.header.get_data_shape())
    hdrImg = img.get_header() #Get header infos of the image
    hdrMask= img.get_header() #Get the header of the mask
    VoxelSizeImg=hdrImg['pixdim'] #Get Pixel Dimension of the image
    VoxelSizeMask=hdrMask['pixdim']
    #################################
    #DEBUGGING PRINTS OF ORIGINAL VOXEL SIZE OF MASK AND BRAIN IMAGE
    #print("Original Voxel SizeImg")
    #print(hdrImg['pixdim'][1:4])
    #print("Original Voxel SizeMask")
    #print(hdrMask['pixdim'][1:4])
    #######################
    #Create an affine matrix that will reduce the voxel dimension of my image of the desired ratio
    Target_affine_Img = np.diag((VoxelSizeImg[1]*DownSampleValue, VoxelSizeImg[2]*DownSampleValue, VoxelSizeImg[3]*DownSampleValue))
    #Target_affine_Img=Target_affine_Img*target_affineStandard
    Target_affine_Mask = np.diag((VoxelSizeMask[1]*DownSampleValue, VoxelSizeMask[2]*DownSampleValue, VoxelSizeMask[3]*DownSampleValue))
    #print(Target_affine_Mask)
    #Use the function resample_img to downsample voxel size
    #The documentation for it can be found here http://nilearn.github.io/manipulating_images/manipulating_images.html
    DownSampledImage=resample_img(img, target_affine=Target_affine_Img)
    DownSampledMask=resample_img(mask,target_affine=Target_affine_Mask)
    #Now multiply Brain Image and Mask
    #print(DownSampledImage.get_header()['pixdim'])
    #img_data = img.get_data()
    NewCroppedImage = (np.asfarray(DownSampledImage.get_data()))*(np.asfarray(DownSampledMask.get_data()))
    value=VoxelSizeImg[1]*DownSampleValue
    affine = np.diag([value,value,value,value])
    Nuova = nib.Nifti1Image(NewCroppedImage,affine)
    #This function crop the image considering the best bounding box around it
    #It leaves the biggest amount of ones in the image itself considering
    #Crop=crop_img(Nuova,32)
    print(Nuova.header.get_data_shape())
    Crop=crop_imgOriginal(Nuova,32)
    print(Crop.header.get_data_shape())
    img_ = np.array(Crop.dataobj)
    # A questo punto posso prendere il valore della shape della immagine croppedNewImage
    # Se non e' divisibile per il valore desiderato
    # Allora posso considerare il valore del resto per quel numero per il valore desiderato
    # Sottraggo il resto al valore iniziale
    # Ottengo il valore desiderato che e' divisibile (per esempio qui faccio le 3 dimensioni 96,128,96)
    return(img_)
