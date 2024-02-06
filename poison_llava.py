import sys
import argparse
import os
import gc

import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
import copy
from torchvision.transforms.functional import InterpolationMode

# diff augmentation
# import kornia
from augmentation_zoo import *

# llava
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

def parse_args():
    parser = argparse.ArgumentParser(description="Poisoning")

    parser.add_argument("--task_data_pth", default='data/task_data/Biden_base_Trump_target', help='task_data_pth folder contains base_train and target_train folders for constructing poison images') 
    parser.add_argument("--poison_save_pth", default='data/poisons/llava/Biden_base_Trump_target', help='Output path for saving pure poison images & original captions') 

    parser.add_argument("--iter_attack", type=int, default=4000)
    parser.add_argument("--lr_attack", type=float, default=0.2)

    parser.add_argument("--diff_aug_specify", type=str, default=None, help='if None, using the default diff_aug of the VLM')

    parser.add_argument("--batch_size", type=int, default=60, help='batch size for running the PGD attack. Modify it according to your GPU memory') 

    args = parser.parse_args()

    if args.diff_aug_specify == "None":
      args.diff_aug_specify = None

    return args

############ model-specific ############
def get_image_encoder_llava():
      '''
      Return: the image encoder, image processor and the data augmention used during training

      image_processor is only for sanity check in test_attack_efficacy()
      diff_aug will be used in crafting adversarial examples
      '''
      model_path = "liuhaotian/llava-v1.5-7b"
      tokenizer, model, image_processor, context_len = load_pretrained_model(
      model_path=model_path,
      model_base=None,
      model_name=get_model_name_from_path(model_path)
      )

      vision_model = copy.deepcopy(model.model.vision_tower); vision_model.eval()
      # In llava, the forward function of CLIP is wrapped with torch.no_grad, which we get rid of below
      image_encoder_ = vision_model.forward.__wrapped__
      image_encoder = lambda x: image_encoder_(vision_model, x)

      # delete the model (including LLM) to save memory
      del model
      gc.collect(); torch.cuda.empty_cache()

      diff_aug = None

      img_size = 336

      return image_encoder, image_processor, diff_aug, img_size

############ model-agnostic ############
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def load_image(image_path, show_image=False):
    img = Image.open(image_path).convert('RGB')
    if show_image:
        plt.imshow(img)
        plt.show()
    return img

def load_image_tensors(task_data_pth,img_size):
    '''
    Input:
    task_data_pth needs to contain two subfolders: base_train and target_train;
    task_data_pth/base_train needs to contain cap.json caption file

    img_size is the size of image for the VLM model. used for resizing.

    Output: image tensors from base_train, target_train 
    '''
    with open(os.path.join(task_data_pth,'base_train','cap.json')) as file:    
        base_train_cap = json.load(file)
    num_total = len(base_train_cap['annotations'])

    images_base = []
    images_target = []

    resize_fn = transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                )

    for i in range(num_total):
        image_id = base_train_cap['annotations'][i]['image_id']
        image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.png')
        image_target_pth = os.path.join(task_data_pth, 'target_train', f'{image_id}.png')

        images_base.append(transforms.ToTensor()(resize_fn(load_image(image_base_pth))).unsqueeze(0)) 
        images_target.append(transforms.ToTensor()(resize_fn(load_image(image_target_pth))).unsqueeze(0)) 

    images_base = torch.cat(images_base, axis=0)
    images_target = torch.cat(images_target, axis=0)
    print(f'Finishing loading {num_total} pairs of base and target images for poisoning, size={images_base.size()}')

    return images_base, images_target

class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_base, images_target):
        '''
        both input image are tensors with (num_example, 3, h, w)
        This dataset be used to construct dataloader for batching
        '''
        super().__init__()
        assert images_base.shape[0] == images_target.shape[0]
        self.images_base = images_base
        self.images_target = images_target

    def __len__(self):
        return self.images_base.shape[0]

    def __getitem__(self, index):
        return self.images_base[index], self.images_target[index]

def embedding_attack_Linf(image_encoder, image_base, image_victim, emb_dist, \
                     iters=100, lr=1/255, eps=8/255, diff_aug=None, resume_X_adv=None):
      '''
      optimizing x_adv to minimize emb_dist( img_embed of x_adv, img_embed of image_victim ) within Lp constraint using PGD

      image_encoder: the image embedding function (e.g. CLIP, EVA)
      image_base, image_victim: images BEFORE normalization, between [0,1]
      emb_dist: the distance metrics for vision embedding (such as L2): take a batch of bs image pairs as input, \
            and output EACH of pair-wise distances of the whole batch (size = [bs])

      eps: for Lp constraint
      lr: the step size. The update is grad.sign * lr
      diff_aug: using differentiable augmentation, e.g. RandomResizeCrop
      resume_X_adv: None or an initialization for X_adv

      return: X_adv between [0,1]
      '''
      assert len(image_base.size()) == len(image_victim.size()) and len(image_base.size()) == 4, 'image size length should be 4'
      assert image_base.size(0) == image_victim.size(0), 'image_base and image_victim contain different number of images'
      bs = image_base.size(0)
      device = image_base.device

      with torch.no_grad():
            embedding_targets = image_encoder(normalize(image_victim))

      X_adv = image_base.clone().detach() + (torch.rand(*image_base.shape)*2*eps-eps).to(device)
      if resume_X_adv is not None:
            print('Resuming from a given X_adv')
            X_adv = resume_X_adv.clone().detach()
      X_adv.data = X_adv.data.clamp(0,1)
      X_adv.requires_grad_(True) 

      optimizer = optim.SGD([X_adv], lr=lr)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(iters*0.5)], gamma=0.5)

      loss_best = 1e8 * torch.ones(bs).to(device)
      X_adv_best = resume_X_adv.clone().detach() if resume_X_adv is not None else torch.rand(*image_base.shape).to(device)

      for i in tqdm(range(iters)):
      # for i in range(iters):
            if diff_aug is not None:
                  # NOTE: using differentiable randomresizedcrop here 
                  X_adv_input_to_model = normalize(diff_aug(X_adv))
            else:
                  X_adv_input_to_model = normalize(X_adv)
            loss = emb_dist(image_encoder(X_adv_input_to_model), embedding_targets) # length = bs

            if i% max(int(iters/1000),1) == 0:
                  if (loss < loss_best).sum()>0:
                        index = torch.where(loss < loss_best)[0]
                        loss_best[index] = loss.clone().detach()[index].to(loss_best[index].dtype)
                        X_adv_best[index] = X_adv.clone().detach()[index]

            loss = loss.sum() 
            optimizer.zero_grad()
            loss.backward()

            if i% max(int(iters/20),1) == 0:
                  print('Iter :{} loss:{:.4f}, lr * 255:{:.4f}'.format(i,loss.item()/bs, scheduler.get_last_lr()[0]*255))

            # Linf sign update
            X_adv.grad = torch.sign(X_adv.grad)
            optimizer.step()
            scheduler.step()
            X_adv.data = torch.minimum(torch.maximum(X_adv, image_base - eps), image_base + eps) 
            X_adv.data = X_adv.data.clamp(0,1)     
            X_adv.grad = None  

            if torch.isnan(loss):
                  print('Encounter nan loss at iteration {}'.format(i))
                  break                 

      with torch.no_grad():
            if diff_aug:
                  print('Using diff_aug')
                  X_adv_best_input_to_model = normalize(diff_aug(X_adv_best))
            else:
                  print('Not using diff_aug')
                  X_adv_best_input_to_model = normalize(X_adv_best)
            loss = emb_dist(image_encoder(X_adv_best_input_to_model), embedding_targets)
            # print('Best Total loss vector:{}'.format(loss))
            print('Best Total loss:{:.4f}'.format(loss.mean().item()))

      return X_adv_best, loss.detach()

def L2_norm(a,b):
      '''
      a,b: batched image/representation tensors
      '''
      assert a.size(0) == b.size(0), 'two inputs contain different number of examples'
      bs = a.size(0)

      dist_vec = (a-b).view(bs,-1).norm(p=2, dim=1)

      return dist_vec

def save_poison_data(images_to_save, caption_pth, save_path):
      '''
      Save the pure poison data set as the same folder format as cc_sbu_align

      Input:
      images_to_save: a batch of image tensors (perturbed base_train images)
      caption_pth: json file path of captions for the unpoisoned images (base_train captions)
      save_path: path for saving poisoned images and original captions. 
      need to save to png, not jpeg.
      '''
      assert len(images_to_save.size()) == 4, 'images_to_save should be a batch of image tensors, 4 dimension'

      with open(caption_pth) as file:    
            cap = json.load(file)
      num_total = len(cap['annotations'])
      assert images_to_save.size(0) == num_total, 'numbers of images and captions are different'

      # save image using the original image_id
      for i in range(num_total):
            image_id = cap['annotations'][i]['image_id']
            img_pth = os.path.join(save_path, 'image', '{}.png'.format(image_id))
            save_image(images_to_save[i],img_pth)

            # rename to .jpg
            img_pth_jpg = os.path.join(save_path, 'image', '{}.jpg'.format(image_id))
            os.rename(img_pth,img_pth_jpg)

      # copy the json file
      shutil.copyfile(caption_pth, os.path.join(save_path,'cap.json'))

      print('Finished saving the pure poison data to {}'.format(save_path))

def test_attack_efficacy(image_encoder, image_processor, task_data_pth, poison_data_pth, img_size, sample_num=20):
      '''
      Sanity check after crafting poison model
      
      Reload image_base, image_target and image_poison from jpg
      Go through image processor, and check the relative distance in the image embedding space
      sample_num: only compute statistics for the first sample_num image triples and then take the average

      Output: will print averaged latent_dist(image_base,image_target) and latent_dist(image_poison,image_target)
      also output the pixel distance between base and poison images

      NOTE: image_processor includes data augmentation. However, when using differantial jpeg during creating poison image,
      the image_processor will not include jpeg operation. 
      '''
      # RGB image
      images_base, images_target = [], []
      images_poison = []

      # load data
      with open(os.path.join(poison_data_pth,'cap.json')) as file:    
            cap = json.load(file)
      num_total = len(cap['annotations'])

      for i in range(num_total):
            image_id = cap['annotations'][i]['image_id']

            image_base_pth = os.path.join(task_data_pth, 'base_train', f'{image_id}.png')
            image_target_pth = os.path.join(task_data_pth, 'target_train', f'{image_id}.png')
            image_poison_pth = os.path.join(poison_data_pth, 'image', f'{image_id}.jpg')

            images_base.append((load_image(image_base_pth)))
            images_target.append((load_image(image_target_pth)))
            images_poison.append((load_image(image_poison_pth)))

            if i >= sample_num:
                  break

      resize_fn = transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                )

      # compute embedding distance
      dist_base_target_list = []
      dist_poison_target_list = []
      pixel_dist_base_poison = [] # Linf distance in pixel space
      for i in range(len(images_base)):
            image_base, image_target, image_poison = images_base[i], images_target[i], images_poison[i]

            emb_base = image_encoder( torch.from_numpy(image_processor(image_base)['pixel_values'][0]).cuda().unsqueeze(0) )
            emb_target = image_encoder( torch.from_numpy(image_processor(image_target)['pixel_values'][0]).cuda().unsqueeze(0) )
            emb_poison = image_encoder( torch.from_numpy(image_processor(image_poison)['pixel_values'][0]).cuda().unsqueeze(0) )

            dist_base_target_list.append( (emb_base - emb_target).norm().item() )
            dist_poison_target_list.append( (emb_poison - emb_target).norm().item() )
            pixel_dist_base_poison.append( torch.norm(transforms.ToTensor()(resize_fn(image_base)) - transforms.ToTensor()(image_poison), float('inf')) )

      dist_base_target_list = torch.Tensor(dist_base_target_list)
      dist_poison_target_list = torch.Tensor(dist_poison_target_list)
      pixel_dist_base_poison = torch.Tensor(pixel_dist_base_poison)

      print('\n Sanity check of the optimization, considering image loading and image processor')
      print(f'>>> ratio betwen dist_base_target and dist_poison_target:\n{dist_base_target_list/dist_poison_target_list}')
      print(f'ratio mean: {(dist_base_target_list/dist_poison_target_list).mean()}')
      print(f'>>> Max Linf pixel distance * 255 between base and poison: {(pixel_dist_base_poison*255).max()}')

      return 

if __name__ == "__main__":
      args = parse_args()

      if os.path.exists(args.poison_save_pth):
            raise ValueError('{} already exists for saving pure poisoned data. Delete it or choose another path!'.format(args.poison_save_pth))
      else:
            os.makedirs(os.path.join(args.poison_save_pth,'image')) 
      print(f'Poisong images will be saved to {args.poison_save_pth}')
      print(f'iter_attack {args.iter_attack}, lr_attack {args.lr_attack}')
            

      ###### model preparation ######
      image_encoder, image_processor, diff_aug, img_size = get_image_encoder_llava()
      if args.diff_aug_specify is not None:
            diff_aug = get_image_augmentation(augmentation_name=args.diff_aug_specify, image_size=img_size)
      else:
            print('Using default diff_aug')

      ###### data preparation ######
      images_base, images_target = load_image_tensors(args.task_data_pth,img_size)
      dataset_pair = PairedImageDataset(images_base=images_base, images_target=images_target)
      dataloader_pair = torch.utils.data.DataLoader(dataset_pair, batch_size=args.batch_size, shuffle=False)

      ###### Running attack optimization ######
      X_adv_list = []
      loss_attack_list = []
      for i, (image_base, image_victim) in enumerate(dataloader_pair):
            # if i == 1:
            #       break
            print('batch_id = ',i)
            image_base, image_victim = image_base.cuda(), image_victim.cuda()
            X_adv, loss_attack = embedding_attack_Linf(image_encoder=image_encoder, image_base=image_base, image_victim=image_victim, emb_dist=L2_norm, \
                        iters=args.iter_attack, lr=args.lr_attack/255, eps=8/255, diff_aug=diff_aug, resume_X_adv=None)
            
            X_adv_list.append(X_adv)
            loss_attack_list.append(loss_attack)

      X_adv = torch.cat(X_adv_list,axis=0)
      loss_attack = torch.cat(loss_attack_list,dim=0)

      ###### Saving poison data ######
      save_poison_data(images_to_save=X_adv.cpu(), caption_pth=os.path.join(args.task_data_pth,'base_train','cap.json'), \
                  save_path=args.poison_save_pth)
      # sanity check (taking into consideration of loading images and image processor)
      test_attack_efficacy(image_encoder=image_encoder, image_processor=image_processor, \
                     task_data_pth=args.task_data_pth, poison_data_pth=args.poison_save_pth, img_size=img_size, sample_num=50)
      print(f'Poisong images are saved to {args.poison_save_pth}')