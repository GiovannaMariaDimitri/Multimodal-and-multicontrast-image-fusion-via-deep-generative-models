import numpy as np
from numpy.random import RandomState
from os import listdir
import nibabel as nib
import scipy.ndimage.interpolation as inter
from utils.DownsampleImage import DownSampleMultiplyResizeCrop

class DataLoader:

    def __init__(self,datapath, target_shape, seed = None):
        #self.datapath = '/home/ses88/rds/rds-t2-cs056/ses88/sample_data'
        #self.datapath = '/local/scratch/ses88/sample_data'
        #Load the images
        #PATH PER OLD IMAGE SET di Nicola
        #self.datapath = '/home/gmd43/Documents/DataNicola/HCP_Imgs'
        #PATH PER NEW DATA SET di Nicola (July 2019, repadded and sampled)
        self.datapath=datapath
        #self.mask_datapath = '/home/ses88/rds/rds-t2-cs056/ses88/HCP_07template0_cropped_brain_mask.nii.gz'
        #LOAD THE MASK
        #self.mask_datapath='/home/gmd43/Documents/DataNicola/HCP_07template0_cropped_brain_mask_p_resampled1.4.nii.gz'
        #self.mask_datapath = '/local/scratch/ses88/sample_data/HCP_07template0_cropped_brain_mask.nii.gz'
        self.target_shape = target_shape
        self.seed = seed
        self.mris = self.split_filenames()


    def shuffle_list (self, _list):
       p = RandomState(self.seed).permutation(len(_list))
       _list = [_list[i] for i in p]
       return _list


    def split_filenames (self):
        files = listdir(self.datapath)
        mris = sorted([_file for _file in files if _file [-2:] == 'gz'])
	#mris = sorted([_file for _file in files if _file [-2:] == 'gz'])

        #pd_files = sorted(listdir(self.pd_datapath))[1:]
        #pd = {}


        #self.num_test_samples = 40
        #control_files = self.shuffle_list(control_files)
        #pd_files = self.shuffle_list(pd_files)

        #pd['test'] = pd_files[:self.num_test_samples]
        #controls['test'] = control_files[:self.num_test_samples]

        #pd['train'] = pd_files[self.num_test_samples:]
        #controls['train'] = control_files[self.num_test_samples:]


        return mris


    def get_data (self):
         mri_volumes = []
         #mask = np.asarray(nib.load(self.mask_datapath).dataobj)
         for mri_name in self.mris:
             #mask_path = self.mask_datapath
             #img_path =self.datapath + '/' + mri_name
             proxy_image = nib.load(self.datapath + '/' + mri_name)


             image=np.asarray(proxy_image.dataobj)

            #Nel nuovo DATASET che mi ha dato Nicola non devo moltiplicare per la maschera perche'
            #ha gia' fatto tutto lui
             #image = np.asarray(proxy_image.dataobj) * mask


             #image = DownSampleMultiplyResizeCrop(img_path, mask_path, 2, 32)
             #proxy_image = nib.load(self.datapath + '/' + mri_name)
             #image = np.asarray(proxy_image.dataobj)

             #zoom_ratio = 256/311.
             #original dimensionality of image (260, 311, 228)
             #image = inter.zoom(image, zoom_ratio) #reshape to 256, 256, 192
             #pad_x = int((256 - image.shape[0])/2)
             #pad_y = 0
             #pad_z = int((192 - image.shape[2])/2)
             #padded_image = np.zeros((256, 256, 192))
             #padded_image[pad_x:256-pad_x, :, pad_z:192-pad_z] = image
             #Maintain original aspect ratio
             #Zoom to smaller size -> pad with 0s to get dimensions divisible by 2
             #padded_image = image[2:258, 30:286, 19: 211]
             mri_volumes.append(np.asarray(np.expand_dims(image, axis = -1)))

         return np.asarray(mri_volumes).astype('float32')




    def get_mri (self):
        mris = self.get_data()
        return mris
