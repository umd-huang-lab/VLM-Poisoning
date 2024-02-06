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

![](Figures_Github/Demo.png)


Method:
![](Figures_Github/PoisonMethod.png)

## Environment

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

## Data preparation

### Crafting Poison
For text, we have already crafted and include it in the data folder.


### Training Models

To train LLaVA-1.5 on poisoned training data, modify `train_llava_lora.sh` and run `bash train_llava_lora.sh`. 