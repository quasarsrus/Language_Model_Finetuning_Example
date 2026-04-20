<div align="center">

# Language_Model_Tuning_Example

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Fine-tuning of HuggingFace's SmolLM-135M causal language model on the Tiny Shakespeare dataset.
The goal is to adapt the base model, pretrained on general web text, to generate
Shakespeare-style text. Training is managed with PyTorch Lightning and Hydra for configuration.

The project supports both full fine-tuning and parameter-efficient fine-tuning (LoRA) via the
PEFT library, and evaluates models using perplexity and ROUGE-L on a held-out test set.

The dataset can be obtained in two ways:
1) https://huggingface.co/datasets/Trelis/tiny-shakespeare
2) https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The first way is a cleaner HuggingFace download from "https://huggingface.co/datasets/Trelis/tiny-shakespeare".
This dataset is already split into train and test sets and is ready for tokenisation.

The second way is by downloading it directly from "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt".
The splits are currently performed as 80:10:10 (Train: Validation: Test), which can be adjusted in the self.setup() method.
After the split, once the text is wrapped using HuggingFace's Dataset class, the rest of the process becomes the same as above.

If an implementation is done using the first type of dataset, we will use the '<=====' symbol to identify it. Similarly,
'===>' will be used for the second one. Commenting and uncommenting appropriately will yield the desired result (in src/data/tinyshakespeare_datamodule.py).


## Installation

uv sync
un run pre-commit install

## How to run

Evaluate base pretrained model (no fine-tuning)
```bash
uv run src/train.py train=False test=True ckpt_path=null
```

Train model with default configuration

```bash
# train on CPU
uv run src/train.py trainer=cpu

# train on GPU
uv run src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
uv run src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
uv run src/train.py trainer.max_epochs=20 data.batch_size=64
```

Train with LoRA
```bash
uv run src/train.py experiment=tiny_shakespeare model.use_lora=true
```

Evaluate fine-tuned checkpoint
```bash
uv run src/eval.py ckpt_path="path/to/model/checkpoint/file.ckpt"
```
