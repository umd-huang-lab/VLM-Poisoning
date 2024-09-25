<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> <em>Shadowcast</em>: Stealthy Data Poisoning Attacks <br>Against Vision-Language Models </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://yuancheng-xu.github.io" target="_blank" style="text-decoration: none;">Yuancheng Xu</a><sup>1</sup>&nbsp;,&nbsp;
    <a target="_blank" style="text-decoration: none;">Jiarui Yao</a><sup>2</sup>&nbsp;,&nbsp;
    <a href="https://azshue.github.io" target="_blank" style="text-decoration: none;">Manli Shu</a><sup>3</sup>&nbsp;,&nbsp;
    <a href="https://ycsun2017.github.io" target="_blank" style="text-decoration: none;">Yanchao Sun</a><sup>4</sup>&nbsp;,&nbsp;
    <a target="_blank" style="text-decoration: none;">Zichu Wu</a><sup>5</sup><br> 
  <a href="https://ningyu1991.github.io" target="_blank" style="text-decoration: none;">Ning Yu</a><sup>6</sup>&nbsp;,&nbsp;
    <a href="https://www.cs.umd.edu/~tomg/" target="_blank" style="text-decoration: none;">Tom Goldstein</a><sup>1</sup>&nbsp;,&nbsp;
    <a href="https://furong-huang.com" target="_blank" style="text-decoration: none;">Furong Huang</a><sup>1</sup>&nbsp;&nbsp; 
    <br/> 
<sup>1</sup>University of Maryland, College Park&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>University of Illinois Urbana-Champaign<br>
&nbsp;&nbsp;&nbsp;&nbsp;<sup>3</sup>Salesforce Research &nbsp;&nbsp;&nbsp;&nbsp;<sup>4</sup>Apple<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;<sup>5</sup>University of Waterloo &nbsp;&nbsp;&nbsp;&nbsp;<sup>6</sup>Netflix Eyeline Studios<br/> 
<br>Neurips, 2024</span>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2402.06659" target="_blank" style="text-decoration: none;">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://vlm-poison.github.io" target="_blank" style="text-decoration: none;">Project Page</a>
</b>
</p>

---

**Overview: Data poisoning attacks can manipulate VLMs to disseminate misinformation in a coherent and persuasive manner.**

Responses of the clean and poisoned LLaVA-1.5 models. The poison samples are crafted using a different VLM, MiniGPT-v2.

![](Figures_Github/Demo.png)


**Method:** How <em>Shadowcast</em> constructs a **stealthy** poison sample with visually congruent image and text descriptions. Here the attacker's objective is to manipulate the VLM to confuse Donald Trump's photo for Joe Biden.

![](Figures_Github/PoisonMethod.png)

Below we provide instructions on how to repeat our experiments. We will release more code soon. Stay tuned!
# Environment

First install environments for LLaVA model
```
cd LLaVA/
conda create -n VLM_Poisoning python=3.10 -y
conda activate VLM_Poisoning
pip install --upgrade pip # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
conda install -c conda-forge cudatoolkit-dev -y
pip install flash-attn --no-build-isolation 
```

Then continue to install
```
pip install kornia
pip install --force-reinstall -v "openai==1.3.1"
pip install -U accelerate
```

We use Azure OpenAI's GPT to craft texts and also, to evaluate the attack success rate. To use Azure OpenAI's GPT, you need to provide the the key and endpoint (e.g., in `~/.bashrc`) as follows.
```
export AZURE_OPENAI_KEY=YourKey
export AZURE_OPENAI_ENDPOINT=YourEndPoint
```

> Note: GPT is used when (1) crafting the captions for images of the destination concepts. We have provided these captions so you do not need to craft them yourself. (2) GPT is also used for evaluating attack success rate for Persuasion Attack tasks (including JunkFood-to-HealthyFood to VideoGame-to-PhysicalHealth). Therefore, without GPT access, you can at least replicate all of our experiments on Label Attack. 


# Data preparation

You can download the data [here](https://drive.google.com/file/d/1kuptRNTe4t_1Sbx-emMl4AlHNrQYbEw0/view?usp=share_link) and unzip it. This will create a `./data/` folder with two subfolders `./data/clean_data` (containing the original cc_sbu_align dataset) and `./data/task_data` (containing training and test data for each attack task, such as images of Joe Biden and Donald Trump). 

> Terminology: base & target image. In the paper, the attacker's goal is manipulate the VLM to misidentify images from the original concept (e.g., Donald Trump) to the destination concept (e.g., Joe Biden). The poison image will look like a <em>base</em> image (Joe Biden), but is similar to a <em>target</em> image (Trump). Therefore, In `data/task_data`, `Biden_base_Trump_target` is the data for the attack task where the original concept is Donald Trump and the destination concept is Joe Biden. 
> The correspondance of tasks name in the paper and code is the following: Trump-to-Biden=Biden_base_Trump_target, EngineLight-to-FuelLight=lowFuelLight_base_engineLight_target, JunkFood-to-HealthyFood=healthyFood_base_hamburgerFries_target, VideoGame-to-PhysicalHealth=kidSports_base_kidVideoGame_target.

# Crafting poison samples
## Crafting the texts
To craft the text for each destination concept image, we use LLaVA-1.5 to generate the caption, which is then refined by GPT-3.5-Turbo. These generated texts are provided in, e.g., `data/task_data/Biden_base_Trump_target/base_train/cap.json`. These texts will also be the texts in the poison samples.

## Crafting poison images

Run `bash poison_llava.sh`. Modify the `--batch_size` according to your GPU memory. Modify `task_name` for different attack tasks. Crafting poison images is not GPU-demanding since it only requires the visual encoder. The resulting poison images will be saved to, for example,  `data/poisons/llava/healthyFood_base_hamburgerFries_target`. 

# Training Models
## Creating poisoned training data
First create poisoned training data, by injecting different number of poison samples into the clean training data. To do this, run 

`python prepare_training_data_w_poison.py --model_name llava --seed 0 --task_name Biden_base_Trump_target`. 

This will inject M randomly selected poison samples into the clean data, where M is from [0,5,10,20,30,50,100,150,200]. The resulting poisoned data will be saved to, e.g., `data/poisoned_training_data/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0.json`

## Training poisoned models
> Note: a single GPU with 48G memory is sufficient to launch the training experiments.

To train LLaVA-1.5 on poisoned training data, modify `train_llava_lora.sh` (such as `task_name` and  `num_poison_list`) and run it. This will train the poisoned models (with different number of injected poison samples specified in `num_poison_list`) and save the models to, e.g, `checkpoints/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0`.

# Evaluation

## Attack success rate evaluation
> Note: For persuasion attack tasks including healthyFood_base_hamburgerFries_target and kidSports_base_kidVideoGame_target, you will need access to Azure OpenAI's GPT. 

Modify  `eval_poison_llava.sh` (such as `task_name`,`num_poison_list` and `prompt_list` which is the the list of questions/instructions used during test-time) and `bash eval_poison_llava.sh`.

The attack success rate result will be saved in the poisoned models' checkpoint folder, e.g., `checkpoints/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0/eval/eval_poison.log`. 
The poisoned VLMs' generated responses are also saved to, for example, `checkpoints/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0/eval/generation.json`. 

## Benchmark evaluation 
First, download the dataset according to the official guide of LLaVA [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). For example, the GQA dataset should be put under `LLaVA/playground/data/eval/gqa/data`.

Then, modify and run `bash benchmark/benchmark_llava_gqa.sh` and `bash benchmark/benchmark_llava_vizwiz.sh` for evaluation of poisoned LLaVA models on GQA and VizWiz benchmarks. 
The results will be saved to, for example, `checkpoints/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0/eval/gqa/result.log`.

# Comments

Stay tuned! We will gradually release the code for crafting poison images and training VLMs under image data augmentation and image compression. 

# Citation
```
@article{xu2024shadowcast,
  title={Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models},
  author={Xu, Yuancheng and Yao, Jiarui and Shu, Manli and Sun, Yanchao and Wu, Zichu and Yu, Ning and Goldstein, Tom and Huang, Furong},
  journal={arXiv preprint arXiv:2402.06659},
  year={2024}
}
```
