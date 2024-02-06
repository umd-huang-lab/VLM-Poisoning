import sys
# diff augmentation
import kornia
# diff jpeg
sys.path.append("DiffJPEG/")
from DiffJPEG import DiffJPEG

augmentation_zoo_list=['lavisCLIP','jpeg']

def getAug_lavisCLIP(image_size):
      '''
      augmentation used in xinstructBLIP
      according to https://github.com/salesforce/LAVIS/blob/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc/lavis/processors/clip_processors.py#L20
      '''
      return kornia.augmentation.RandomResizedCrop(
      size=(image_size, image_size), scale=(0.9, 1.0), ratio=(0.75, 1.3333333333333333),
      resample="BICUBIC", align_corners=True, cropping_mode="resample", same_on_batch=False, keepdim=False, p=1.0)

def get_image_augmentation(augmentation_name, image_size):
      assert augmentation_name in augmentation_zoo_list, f'{augmentation_name} is not implemented'

      if augmentation_name == 'lavisCLIP':
            print('Using lavisCLIP as the image augmentation method. lavisCLIP is used in xInstructBLIP')
            return getAug_lavisCLIP(image_size=image_size)

      if augmentation_name == 'jpeg':
            print('Using diff jpeg as the image augmentation method')
            return DiffJPEG(height=image_size, width=image_size, differentiable=True, quality=75).cuda()
