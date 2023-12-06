# Breaking the Language Barrier: Improving Cross-Lingual Reasoning with Structured Self-Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper: "[Breaking the Language Barrier: Improving Cross-Lingual Reasoning with Structured Self-Attention](https://arxiv.org/abs/2310.15258)"  [EMNLP 2023 - Findings]


## Requirements
We recommend using a conda environment for running the scripts.
You can run the following commands to create the conda environment (assuming CUDA11.3):
```bash
conda create -n cross-lingual-attention python=3.10.9
conda activate cross-lingual-attention
pip install -r requirements.txt
conda install faiss-cpu -c pytorch
```

## Data
Translations of RuleTaker and LeapOfThought (LoT) datasets can be found in the `data` folder. For the LoT dataset, the files start with  **"randomized"** are the modified version of the data we used for our experiments (randomly negate 50% of statements). For more information on this dataset, please check Appendix A.2 of the paper.

## Usage
In all the following experiments, you can either pass the arguments directly to the python script or specify the arguments in a `JSON` file and pass the file path to the script.

#### Fine-tuning a model for RuleTaker or LoT datasets:

```shell
python standard_finetuning.py \
	--output_dir ruletaker_finetuning \
    --data_base_dir data \ # Assuming each language has a folder inside this
	--model_type mbert \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--evaluate_during_training \
    --dataset_name ruletaker \
    --rule_taker_depth_level 0 \
	--num_train_epochs 4 \
    --learning_rate 1e-5 \
    --save_strategy epoch \
    --evaluation_strategy steps\
    --logging_steps 1000 \
    --train_language en \
    --second_train_language fr \ # In case of fine-tuning on two datasets
	--overwrite_output_dir \
	--seed 57
```

#### Fine-tuning a model using the proposed cross-lingual-aware attention mechanism (for RuleTaker or LoT datasets):


```shell
python finetuning_sep_cross_lingual_attention.py \
	--output_dir ruletaker_finetuning_cross_query \
    --data_base_dir data \ # assuming each language has a folder inside this
	--model_type mbert \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--evaluate_during_training \
    --dataset_name ruletaker \
    --rule_taker_depth_level 0 \
    --bitfit \
	--num_train_epochs 35 \
    --learning_rate 4e-4 \
    --warmup_ratio 0.1 \
    --save_strategy epoch \
    --evaluation_strategy steps\
    --logging_steps 1000 \
    --mono_alpha  1.0 \
    --cross_mono_alpha  0.3 \
    --cross_alpha  0.3 \
    --mono_eval_alpha  1.0 \
    --cross_mono_eval_alpha  0.0 \
    --cross_eval_alpha  0.0 \
    --language1 en \
    --language2 en-fr \
    --load_query_pretrained \
	--overwrite_output_dir \
	--seed 57
```


#### Pre-training the cross-lingual query matrix:

```shell
python pretrain_sep_cross_lingual_attention.py \
	--output_dir pretrain_cross_lingual_query \
    --model_type mbert \
    --do_train \
	--do_eval \
    --mlm \
    --evaluate_during_training \ 
    --pad_to_max_length \
    --dataset_name xnli \
    --data_language_pairs en-fr;en-de;en-es;en-ru;en-ar;en-zh \
    --mono_alpha  0.0 \
    --cross_alpha  0.0 \
    --mono_eval_alpha  0.0 \
    --cross_eval_alpha  0.0 \
    --per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_steps 500000 \
    --logging_steps 5000 \
    --save_steps 10000 \
    --freeze_the_rest \
	--overwrite_output_dir \
	--seed 57
```
### Notes:
- Multi-GPU is currently not supprted for the scripts that use the proposed cross-lingual query matrix (i.e., `pretrain_sep_cross_lingual_attention.py`, and `finetuning_sep_cross_lingual_attention.py`).

## Citation

If you use this code for your research, please cite our paper:

``` bib
@article{foroutan2023breaking,
  title={Breaking the Language Barrier: Improving Cross-Lingual Reasoning with Structured Self-Attention},
  author={Foroutan, Negar and Banaei, Mohammadreza and Aberer, Karl and Bosselut, Antoine},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  url={https://arxiv.org/abs/2310.15258}
  year={2023}
}

