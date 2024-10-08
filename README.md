# MINERS <img src="assets/pickaxe.png" width="30px">: Multilingual Language Models as Semantic Retrievers
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

⚡ Introducing the **MINERS benchmark**, designed to assess the multilingual LMs' prowess in semantic retrieval tasks, including bitext mining and classification through retrieval-augmented contexts **without fine-tuning**. A comprehensive framework has been developed to evaluate the effectiveness of language models in retrieving samples across over **200 diverse languages**, including low-resource languages in challenging **cross-lingual (XS)** and **code-switching (CS)** settings. The results show that achieving competitive performance with state-of-the-art methods is possible by solely retrieving semantically similar embeddings, without requiring any fine-tuning.

The paper has been accepted at EMNLP 2024 Findings.

<p align="center">
  <img src="assets/pipeline.png" width="100%">
</p>

## Table of Contents

- [Paper](#-paper)
- [Benchmark](#-benchmark)
- [Environment Setup](#-environment-setup)
- [Experiment Logs](#-experiment-logs)
- [Running Experiments](#-running-experiments)
	- [Bitext Retrieval](#bitext-retrieval)
	- [Retrieval-based Classification](#retrieval-based-classification)
	- [ICL Classification](#icl-classification)
- [Aggregating Experiment Results](#-aggregating-experiment-results)
- [Visualizing the Embeddings](#-visualizing-the-embeddings)
- [Models Support](#-models-support)
- [How to Contribute?](#-how-to-contribute)
- [On Progress](#on-progress)

## 📜 Paper 
This is the source code of the paper [[Arxiv]](https://arxiv.org/abs/2406.07424):

This code has been written using PyTorch. If you use any code or datasets from this toolkit in your research, please cite the associated paper.
<pre>
@article{winata2024miners,
  title={MINERS: Multilingual Language Models as Semantic Retrievers},
  author={Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
  journal={arXiv preprint arXiv:2406.07424},
  year={2024}
}
</pre>

## 📊 Benchmark
MINERS comprises **11** datasets: **7** multilingual and **4** code-switching datasets, covering more than **200 languages** and encompassing both parallel and classification formats. Parallel datasets are suited for bitext retrieval as they contain aligned multilingual content, facilitating bitext mining and machine translation tasks. Additionally, the classification datasets cover intent classification, sentiment analysis, and topic classification, which we assess for retrieval-based and ICL classification assignments.
<p align="center">
  <img src="assets/dataset.png" width="70%">
</p>

Our benchmark evaluates LMs on three tasks: bitext retrieval, retrieval-based classification, and ICL classification. The settings include **monolingual (Mono)**, **cross-lingual (XS)**, **code-switching (CS)**, and **cross-lingual code-switching (XS CS)**.
<p align="center">
  <img src="assets/res_bitext_classification.png" width=42%">
  <img src="assets/res_icl_v2.png" width="53%">
</p>

## ⚡ Environment Setup
```
pip install -r requirements.txt
```
If you wish to utilize the APIs or models from OpenAI, Cohere, or Hugging Face, modify the `OPENAI_TOKEN`, `COHERE_TOKEN`, and `HF_TOKEN`. Note that most models on Hugging Face do not require the `HF_TOKEN`, which is specifically intended for the llama and gemma models.

If you wish to use Llama3.1, you need to upgrade the transformers version
```
pip install transformers==4.44.2
```

## 📝 Experiment Logs
If you wish to get all results and prompt examples from our experiments, feel free to download them [here](https://drive.google.com/file/d/1yG4VQDClLAhlyGZNxrnByZbOdU2kaAAR/view?usp=drive_link) (~360MB).

## 🧪 Running Experiments
All experiment results will be stored in the `logs/` directory. You can execute each experiment using the following commands:

### Bitext Retrieval
#### Cross-lingual setting
```
❱❱❱ python bitext.py --src_lang {src_lang} --dataset {dataset} --seed {seed} --cuda --model_checkpoint {model_checkpoint}
❱❱❱ python bitext.py --src_lang de --dataset bucc --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
```

#### Ensemble
The arguments are similar as above, except we use `--model_checkpoints` and `--weights`
```
❱❱❱ python bitext.py --src_lang {src_lang} --dataset {dataset} --seed {seed} --cuda --model_checkpoint {model_checkpoint}
❱❱❱ python bitext.py --src_lang de --dataset bucc --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
```

### Retrieval-based Classification
#### Monolingual setting
```
❱❱❱ python classification.py --dataset {dataset} --seed {seed} --cuda --model_checkpoint {model_checkpoint}
❱❱❱ python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
```

#### Cross-lingual setting
Add `--src_lang` and `--cross` to the command.
```
❱❱❱ python classification.py --src_lang {src_lang} --cross --dataset {dataset} --seed {seed} --cuda --model_checkpoint {model_checkpoint}
❱❱❱ python classification.py --src_lang eng --cross --dataset nusax --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
```

#### Ensemble
The arguments are similar as above, except we use `--model_checkpoints` and `--weights`
```
❱❱❱ python classification.py --dataset {dataset} --seed {seed} --cuda --model_checkpoints {model_checkpoint1} {model_checkpoint2} {...} --weights {weight1} {weight2} {...}
❱❱❱ python classification.py --dataset nusax --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
```

### ICL Classification
#### Monolingual setting
```
❱❱❱ python icl.py --dataset {dataset} --seed 42 --instruction {instruction} --model_checkpoint {model} --gen_model_checkpoint {gen_model_checkpoint}  --cuda --load_in_8bit --k {k}
❱❱❱ python icl.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint meta-llama/Meta-Llama-3-8B-Instruct  --cuda --load_in_8bit --k 1
```

#### Cross-lingual setting
Add `--src_lang` and `--cross` to the command.
```
❱❱❱ python icl.py --src_lang {src_lang} --cross --dataset {dataset} --seed 42 --instruction {instruction} --model_checkpoint {model} --gen_model_checkpoint {gen_model_checkpoint}  --cuda --load_in_8bit --k {k}
❱❱❱ python icl.py --src_lang eng --cross --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint meta-llama/Meta-Llama-3-8B-Instruct  --cuda --load_in_8bit --k 1
```

## 📈 Aggregating Experiment Results
Add `--k` to modify the number of retrieved samples.
```
❱❱❱ python script/aggregate/aggregate_bitext_mining.py --k {k}
❱❱❱ python script/aggregate/aggregate_classification.py --k {k}
❱❱❱ python script/aggregate/aggregate_classification_cross.py --k {k}
❱❱❱ python script/aggregate/aggregate_icl.py --k {k}
❱❱❱ python script/aggregate/aggregate_icl_cross.py --k {k}
❱❱❱ python script/aggregate/aggregate_icl_percentile.py --k {k}
```

## 🏞️ Visualizing the Embeddings
```
❱❱❱ python visualize.py --model_checkpoint {model_checkpoint} --dataset {dataset} --seed {seed} --cuda
❱❱❱ python visualize.py --model_checkpoint sentence-transformers/LaBSE --dataset nusax --seed 42 --cuda
```

### Examples of the visualization by class labels: LaBSE (left) and XLM-R BASE (right)
<img src="assets/scatter_plots/tsne_nusax_LaBSE_class.png" width="35%"> <img src="assets/scatter_plots/tsne_nusax_xlm-roberta-base_class.png" width="35%">

### Examples of the visualization by sample ID: LaBSE (left) and XLM-R BASE (right)
<img src="assets/scatter_plots/tsne_nusax_LaBSE.png" width="35%"> <img src="assets/scatter_plots/tsne_nusax_xlm-roberta-base.png" width="35%">

## 💻 Models Support
Our codebase supports the usage of multiple models for the experiments, providing flexibility for customization beyond the list shown below:
### Encoder LMs and APIs
#### Open-source LMs:
- [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)
- [sentence-transformers/use-cmlm-multilingual](https://huggingface.co/sentence-transformers/use-cmlm-multilingual)
- [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [microsoft/Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)
- [cis-lmu/glot500-base](https://huggingface.co/cis-lmu/glot500-base)
- [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large)

#### Commercial embedding APIs (last tested as of June 2024)
- Cohere-Embedv3
- OpenAI-Embedv3

### Generative LMs:
- BLOOMZ [bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7) [bigscience/bloomz-3b](https://huggingface.co/bigscience/bloomz-3b)
- mT0 [bigscience/mt0-xl](https://huggingface.co/bigscience/mt0-xl)
- XGLM [facebook/xglm-564M](https://huggingface.co/facebook/xglm-564M) [facebook/xglm-2.9B](https://huggingface.co/facebook/xglm-2.9B)
- Aya-23 [CohereForAI/aya-23-8B](https://huggingface.co/CohereForAI/aya-23-8B)
- Aya-101 [CohereForAI/aya-101](https://huggingface.co/CohereForAI/aya-101)
- Gemma 1.1 Instruct [google/gemma-1.1-7b-it](https://huggingface.co/google/gemma-1.1-7b-it)
- Llama 3 8B Instruct [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- Llama 3 8B Instruct [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- GPT models  (last tested as of June 2024)
- Cohere Command R  (last tested as of June 2024)


## 🚀 How to Contribute?
Feel free to create [an issue](https://github.com/gentaiscool/miners/issues/) if you have any questions. And, create [a PR](https://github.com/gentaiscool/miners/pulls) for fixing bugs or adding improvements (i.e., adding new datasets or models). 

If you are interested to create an extension of this work, feel free to reach out to [us](mailto:gentaindrawinata@gmail.com)!

Support our open source effort ⭐

## On Progress
We are improving the code to make it more user-friendly and customizable. We have created a new repository for implementing DistFuse, which is available at [https://github.com/gentaiscool/distfuse/](https://github.com/gentaiscool/distfuse/). You can install it by running `pip install distfuse`. Later, it will be integrated to this repository.
