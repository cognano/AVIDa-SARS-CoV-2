# A SARS-CoV-2 Interaction Dataset and VHH Sequence Corpus for Antibody Language Models

This repository contains the supplementary material accompanying the paper "[A SARS-CoV-2 Interaction Dataset and VHH Sequence Corpus for Antibody Language Models](https://arxiv.org/abs/2405.18749)."
In this paper, we introduced AVIDa-SARS-CoV-2, a labeled dataset of SARS-CoV-2-VHH interactions, and VHHCorpus-2M, which contains over two million VHH sequences, providing novel datasets for the evaluation and pre-training of antibody language models.
The datasets are available at https://datasets.cognanous.com under a CC BY-NC 4.0 license.

<img src="./docs/images/data_generation_overview.png" alt="dataset-generation-overview">

<div style="text-align: center;">
Overview of data generation process for AVIDa-SARS-CoV-2.
</div>

## Table of Contents

- [Environment](#environment)
- [Datasets](#datasets)
  - [Links](#links)
  - [Data Processing](#data-processing)
- [Benchmarks](#benchmarks)
  - [Pre-training](#pre-training)
  - [Fine-tuning](#fine-tuning)
- [Citation](#citation)

## Environment

To get started, clone this repository and run the following command to create a virtual environment.

```bash
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Datasets

### Links

| Dataset          |                                                                                          Links                                                                                           |
| ---------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| VHHCorpus-2M     |      [Hugging Face Hub](https://huggingface.co/datasets/COGNANO/VHHCorpus-2M)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Project Page](https://vhh-corpus.cognanous.com)      |
| AVIDa-SARS-CoV-2 | [Hugging Face Hub](https://huggingface.co/datasets/COGNANO/AVIDa-SARS-CoV-2)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Project Page](https://avida-sars-cov-2.cognanous.com) |

### Data Processing

The code for converting the raw data (FASTQ file) obtained from next-generation sequencing (NGS) into the labeled dataset, AVIDa-SARS-CoV-2, can be found under `./dataset`.
We released the FASTQ files for antigen type "OC43" [here](https://drive.google.com/drive/folders/151Njm6OE9G5m8vyzDcn8w8mWye8ULsYU?usp=sharing) so that the data processing can be reproduced.

First, you need to create a Docker image.

```bash
docker build -t vhh_constructor:latest ./dataset/vhh_constructor
```

After placing the FASTQ files under `dataset/raw/fastq`, execute the following command to output a labeled CSV file.

```bash
bash ./dataset/preprocess.sh
```

## Benchmarks

### Pre-training

VHHBERT is a RoBERTa-based model pre-trained on two million VHH sequences in VHHCorpus-2M.
VHHBERT can be pre-trained with the following commands.

```bash
python benchmarks/pretrain.py --vocab-file "benchmarks/data/vocab_vhhbert.txt" \
  --epochs 20 \
  --batch-size 128 \
  --save-dir "outputs"
```

**Arguments:**

| Argument      | Required | Default   | Description                      |
|---------------|----------|-----------|----------------------------------|
| --vocab-file  | Yes      |           | Path of the vocabulary file      |
| --epochs      | No       | 20        | Number of epochs                 |
| --batch-size  | No       | 128       | Size of mini-batch               |
| --seed        | No       | 123       | Random seed                      |
| --save-dir    | No       | ./saved   | Path of the save directory       |

The pre-trained VHHBERT, released under the MIT License, is available on the [Hugging Face Hub](https://huggingface.co/COGNANO/VHHBERT).

### Fine-tuning

To evaluate the performance of various pre-trained language models for antibody discovery, we defined a binary classification task to predict the binding or non-binding of unknown antibodies to 13 antigens using AVIDa-SARS-CoV-2.
For more information on the benchmarking task, see the [paper](https://arxiv.org/abs/2405.18749).

Fine-tuning of the language models can be performed using the following command.

```bash
python benchmarks/finetune.py --palm-type "VHHBERT" \
  --epochs 30 \
  --batch-size 32 \
  --save-dir "outputs"
```

`palm-type` must be one of the following:
- `VHHBERT`
- `VHHBERT-w/o-PT`
- `AbLang`
- `AntiBERTa2`
- `AntiBERTa2-CSSP`
- `IgBert`
- `ProtBert`
- `ESM-2-150M`
- `ESM-2-650M`

**Arguments:**

| Argument          | Required | Default                                  | Description                          |
| ----------------- | -------- | ---------------------------------------- | ------------------------------------ |
| --palm-type       | No       | VHHBERT                                  | Model name                           |
| --embeddings-file | No       | ./benchmarks/data/antigen_embeddings.pkl | Path of embeddings file for antigens |
| --epochs          | No       | 20                                       | Number of epochs                     |
| --batch-size      | No       | 128                                      | Size of mini-batch                   |
| --seed            | No       | 123                                      | Random seed                          |
| --save-dir        | No       | ./saved                                  | Path of the save directory           |

## Citation

If you use AVIDa-SARS-CoV-2, VHHCorpus-2M, or VHHBERT in your research, please use the following citation.

```bibtex
@inproceedings{tsuruta2024sars,
  title={A {SARS}-{C}o{V}-2 Interaction Dataset and {VHH} Sequence Corpus for Antibody Language Models},
  author={Hirofumi Tsuruta and Hiroyuki Yamazaki and Ryota Maeda and Ryotaro Tamura and Akihiro Imura},
  booktitle={Advances in Neural Information Processing Systems 37},
  year={2024}
}
```
