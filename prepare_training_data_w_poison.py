import sys
import argparse
import os

import json
from distutils.dir_util import copy_tree
import shutil

import random
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training data with poisons")

    parser.add_argument("--model_name", default='llava', choices=['llava', 'miniGPT4v2'],
                        help='VLM to be finetuned') 
    parser.add_argument("--source_model_name", default=None, choices=['llava', 'miniGPT4v2','instructBLIP'],
                        help='VLM to be finetuned') 
    parser.add_argument("--seed", type=int, default=0, help='seed for generating poisons as in the poison folder\'s name')

    parser.add_argument("--task_name", default='Biden_base_Trump_target', choices=['Biden_base_Trump_target', 'lowFuelLight_base_engineLight_target', 'healthyFood_base_hamburgerFries_target', 'kidSports_base_kidVideoGame_target'],
                        help='attack task name') 

    args = parser.parse_args()

    return args

def create_poisoned_training_data(task_name, model_name, source_model_name, num_poison, seed, data_root, clean_data_name):
      '''
      for llava: only need to contruct caption json file. The image path is relative to ./data

      for lavis models (such as minigpt4 and instructBLIP)
            Given output_pth
            when finetuning, set:
            datasets.coco_caption.build_info.images.storage= $output_pth
            datasets.coco_caption.build_info.annotations.train.storage= $output_pth/cap_coco_style.json

            inside output_pth, append data from pure_poison_data_pth using appropriate image_id
      '''

      prompt = "Describe this image in detail."
      assert task_name in ['Biden_base_Trump_target', 'healthyFood_base_hamburgerFries_target', 'kidSports_base_kidVideoGame_target', 'speedLimitSign_base_stopSign_target','lowFuelLight_base_engineLight_target']
      assert model_name in ['llava', 'instructBLIP', 'miniGPT4v2','llava_jpeg','llava_aug_lavisCLIP','llava_aug_lavisCLIP_jpeg','llava_aug_jpeg_jpeg']
      assert source_model_name in ['llava', 'instructBLIP', 'miniGPT4v2', None]
      assert model_name != source_model_name

      if source_model_name is None:
            poison_data_pth = os.path.join(data_root, 'poisons',f'{model_name}', f'{task_name}')
            model_setting = model_name
      else:
            # transfer attack setting
            poison_data_pth = os.path.join(data_root, 'poisons',f'{source_model_name}', f'{task_name}')
            model_setting = f'{source_model_name}_to_{model_name}'
            assert model_name in ['llava', 'instructBLIP', 'miniGPT4v2']

      if 'llava' in model_name:
            clean_data_pth = os.path.join(data_root, 'clean_data', f'{clean_data_name}-llava') 
            output_pth = os.path.join(data_root, 'poisoned_training_data', f'{model_setting}', \
                  f'{clean_data_name}-{task_name}',f'poison_{num_poison}-seed_{seed}.json')
      else:
            clean_data_pth = os.path.join(data_root, 'clean_data', f'{clean_data_name}') 
            output_pth = os.path.join(data_root, 'poisoned_training_data', f'{model_setting}', \
                  f'{clean_data_name}-{task_name}',f'poison_{num_poison}-seed_{seed}/')
            
      print(f'poison_data_pth is {poison_data_pth}')
      print(f'clean_data_pth is {clean_data_pth}')
      print(f'output_pth is {output_pth}')

      if os.path.exists(output_pth):
            raise ValueError(f'{output_pth} already exists for saving poisoned training data. Delete it or choose another path!')
      
      if 'llava' in model_name:
            os.makedirs(os.path.join(data_root, 'poisoned_training_data', f'{model_setting}',f'{clean_data_name}-{task_name}'), exist_ok=True)
            
            # for llava, only need to construct the json file
            with open(os.path.join(clean_data_pth, 'cap.json')) as file:    
                  clean_cap = json.load(file)

            with open(os.path.join(poison_data_pth, 'cap.json')) as file:    
                  poison_cap = json.load(file)

            # update the image pth in clean data json file (relative to data_root)
            for i in range(len(clean_cap)):
                  clean_cap[i]['image'] = os.path.join(f'clean_data/{clean_data_name}-llava',clean_cap[i]['image'])

            # sample poison data
            total_poison_num = len(poison_cap['annotations'])
            assert num_poison <= total_poison_num, f'Total poison size is {total_poison_num}, smaller than specified num_poison {num_poison}'
            random.seed(seed)
            index = random.sample(range(total_poison_num),num_poison)

            # append poison data to the clean data json file
            for i in range(total_poison_num):
                  if i in index:
                        llava_dict = {}
                        img_id = poison_cap['annotations'][i]['image_id']
                        llava_dict["id"] = 'poison-' + img_id
                        if source_model_name is None:
                              llava_dict["image"] = os.path.join(f'poisons/{model_name}/{task_name}','image',f'{img_id}.jpg') 
                        else:
                              llava_dict["image"] = os.path.join(f'poisons/{source_model_name}/{task_name}','image',f'{img_id}.jpg') 
                        llava_dict["conversations"] = [ 
                              {
                                    "from": "human",
                                    "value": f"<image>\n{prompt}"
                              },
                              {
                                    "from": "gpt",
                                    "value": poison_cap['annotations'][i]['caption']
                              },
                        ]

                        clean_cap.append(llava_dict)

            # save clean_cap as the final json file for poisoned training data
            with open(output_pth, 'w', encoding='utf-8') as f:
                  json.dump(clean_cap, f, ensure_ascii=False, indent=4)  

      else:
            '''
            lavis model (minigpt4, instructBLIP)
            '''
            os.makedirs(output_pth, exist_ok=True)

            # create a new copy of clean_data_pth folder to output_poison_pth
            copy_tree(clean_data_pth, output_pth)

            with open(os.path.join(output_pth, 'cap_coco_style.json')) as file:    
                  clean_cap = json.load(file)

            with open(os.path.join(poison_data_pth, 'cap.json')) as file:    
                  poison_cap = json.load(file)

            start_image_id_poison = int( clean_cap[-1]['image'].split('/')[-1].removesuffix('.jpg') ) + 1

            # sample poison data
            total_poison_num = len(poison_cap['annotations'])
            assert num_poison <= total_poison_num, f'Total poison size is {total_poison_num}, smaller than specified num_poison {num_poison}'
            random.seed(seed)
            index = random.sample(range(total_poison_num),num_poison)

            for i in range(total_poison_num):
                  if i in index:
                        # from poison cap
                        image_id_poison = int(poison_cap["annotations"][i]["image_id"])
                        caption_poison = poison_cap["annotations"][i]["caption"]

                        # copy the image from poison folder to output folder
                        shutil.copyfile(os.path.join(poison_data_pth,'image','{}.jpg'.format(image_id_poison)),\
                                    os.path.join(output_pth,'train','{}.jpg'.format(image_id_poison + start_image_id_poison)))
                        
                        # append poison data info to clean data cap
                        clean_cap.append(
                              {
                                    "caption": caption_poison, "image": f"train/{image_id_poison + start_image_id_poison}.jpg", \
                                    "image_id": f"poison_{image_id_poison + start_image_id_poison}"
                              }
                        )

            # save clean_cap as the final json file for poisoned training data
            with open(os.path.join(output_pth, 'cap_coco_style.json'), 'w', encoding='utf-8') as f:
                  json.dump(clean_cap, f, ensure_ascii=False, indent=4)  



if __name__ == "__main__":
      args = parse_args()

      if args.task_name == 'lowFuelLight_base_engineLight_target':
            num_poison_list = [0,5,10,20,30,50,100,150,178]
      else:
            num_poison_list = [0,5,10,20,30,50,100,150,200]

      data_root = './data'
      clean_data_name = 'cc_sbu_align'
      for num_poison in num_poison_list:
            create_poisoned_training_data(task_name=args.task_name, model_name=args.model_name, source_model_name=args.source_model_name, \
                                          num_poison=num_poison, seed=args.seed, data_root=data_root, clean_data_name=clean_data_name)

      print('Done with creating training data with poison samples injected!')