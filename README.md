<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> <em>Shadowcast</em>: Stealthy Data Poisoning Attacks <br>Against Vision-Language Models </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://yuancheng-xu.github.io" target="_blank" style="text-decoration: none;">Yuancheng Xu</a><sup>1</sup>&nbsp;,&nbsp;
    <a target="_blank" style="text-decoration: none;">Jiarui Yao</a><sup>1</sup>&nbsp;,&nbsp;
    <a href="https://azshue.github.io" target="_blank" style="text-decoration: none;">Manli Shu</a><sup>1</sup>&nbsp;,&nbsp;
    <a href="https://ycsun2017.github.io" target="_blank" style="text-decoration: none;">Yanchao Sun</a><sup>2</sup>&nbsp;,&nbsp;
    <a target="_blank" style="text-decoration: none;">Zichu Wu</a><sup>3</sup><br> 
  <a href="https://ningyu1991.github.io" target="_blank" style="text-decoration: none;">Ning Yu</a><sup>4</sup>&nbsp;,&nbsp;
    <a href="https://www.cs.umd.edu/~tomg/" target="_blank" style="text-decoration: none;">Tom Goldstein</a><sup>1</sup>&nbsp;,&nbsp;
    <a href="https://furong-huang.com" target="_blank" style="text-decoration: none;">Furong Huang</a><sup>1</sup>&nbsp;&nbsp; 
    <br/> 
University of Maryland, College Park<sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;JP Morgan AI Research<sup>2</sup>
&nbsp;&nbsp;&nbsp;&nbsp;University of Waterloo<sup>3</sup>
&nbsp;&nbsp;&nbsp;&nbsp;Salesforce Research<sup>4</sup><br/> 
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
pip install flash-attn --no-build-isolation
```

Then continue to install
```
pip install --force-reinstall -v "openai==1.3.1"
pip install kornia
```

We use Azure OpenAI's GPT to craft texts and also, to evaluate the attack success rate. To use Azure OpenAI's GPT, you need to provide the the key and endpoint (e.g., in `~/.bashrc`) as follows
```
export AZURE_OPENAI_KEY=YourKey
export AZURE_OPENAI_ENDPOINT=YourEndPoint
```

# Data preparation

> Terminology: base & target image. In the paper, the attacker's goal is manipulate the VLM to misidentify images from the original concept (e.g., Donald Trump) to the destination concept (e.g., Joe Biden). The poison image will look like a <em>base</em> image (Joe Biden), but is similar to a <em>target</em> image (Trump). Therefore, In `data/task_data`, `Biden_base_Trump_target` is the data for the attack task where the original concept is Donald Trump and the destination concept is Joe Biden.

# Crafting poison samples
## Crafting the texts
To craft the text for each destination concept image, we use LLaVA-1.5 to generate the caption, which is then refined by GPT-3.5-Turbo. The caption is provided in, e.g., `data/task_data/Biden_base_Trump_target/base_train/cap.json`. These texts will also be the texts in the poison samples.

## Crafting poison images

Run `python poison.llava.py`. Modify the `--batch_size` according to your GPU memory. Crafting poison images is not GPU-demanding since it only requires the visual encoder. 

# Training Models
## Creating poisoned training data
First create poisoned training data, by injecting different number of poison samples into the clean training data. To do this, run `python prepare_training_data_w_poison.py --model_name llava --seed 0 --task_name Biden_base_Trump_target`. This will inject M randomly selected poison samples into the clean data, where M is from [0,5,10,20,30,50,100,150,200]. The resulting poisoned data will be saved to, e.g., `data/poisoned_training_data/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0.json`

## Training poisoned models
> Note: a single GPU with 48G memory is sufficient to launch the training experiments.

To train LLaVA-1.5 on poisoned training data, modify `train_llava_lora.sh` and run `bash train_llava_lora.sh`. This will saved the models to, e.g, `checkpoints/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0`.

# Evaluation

## Evaluate attack success rate

Modify and run `eval_poison_llava.sh`. The result will be saved in the poisoned models' checkpoint folder, e.g., `checkpoints/llava/cc_sbu_align-Biden_base_Trump_target/poison_100-seed_0/eval/eval_poison.log`




<!-- # Citation
```
@inproceedings{
xu2023exploring,
title={Exploring and Exploiting Decision Boundary Dynamics for Adversarial Robustness},
author={Yuancheng Xu and Yanchao Sun and Micah Goldblum and Tom Goldstein and Furong Huang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://arxiv.org/abs/2302.03015}
}
``` -->
