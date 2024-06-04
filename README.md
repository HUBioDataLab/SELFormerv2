# SELFORMERv2: A multimodal approach to molecular property prediction problems using SELFIES, 3D graphs and knowledge graph embeddings

## Contents
- [Getting Started](#getting-started)
- [How to Reproduce the Results](#how-to-reproduce-the-results)
    1. [Generating Multimodal Embeddings Using Pre-trained Models](#1-generating-multimodal-embeddings-using-pre-trained-models)
    2. [Training Multimodal Classification Model](#training-multimodal-classification-model)
- [Generating Predictions for New Molecules](#generating-predictions-for-new-molecules)

## Getting Started
**Step 1.** We recommend the Conda platform for installing dependencies. Following the installation of Conda, please create and activate an environment with dependencies as defined below:

```
conda create -n SELFormerv2_env
conda activate SELFormerv2_env
conda env update --file data/requirements.yml
```

**Step 2.** PyTorch Geometric may require additional installation steps. Please refer to the [official documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

**Step 3.** TODO: unimol installation

<br/>

## How to Reproduce the Results

In this section, we provide a step-by-step guide to reproduce the results of the SELFormerv2 model for the BBBP, BACE, and HIV datasets. The following steps are required to generate multimodal embeddings using pre-trained models, and train the multimodal classification model.

### 1. Generating Multimodal Embeddings Using Pre-trained Models

SELFormerv2 utilises multiple molecular features including SELFIES notations, 3D molecular structures, textual descriptions, and knowledge graphs. Each molecular feature is encoded using specialised pre-trained models; namely, SELFormer for SELFIES notations, Uni-Mol for 3D structures, SciBERT for textual descriptions, and DMGI for the knowledge graph. For generating embeddings for each modality, the following steps are required:

**Step 1.** Download the pre-trained SELFormer model (modelO) from [this link](https://drive.google.com/file/d/1zuVAKXCMc-HZHQo9y3Hu5zmQy51FGduI/view?usp=sharing) and place it in the **/data/pretrained_models/** directory. Other pre-trained models will be directly called/downloaded inside the code.

**Step 2.** Download the knowledge graph data from [here](https://drive.google.com/file/d/1u8kg7uzQ-q-osxIvrbJeAsFAbes1TmDF/view?usp=share_link) and extract it to data/knowledge_graph/ directory.

**Step 3.** Run the following command to generate embeddings for each classification dataset (BBBP, BACE, and HIV) using the pre-trained models:

```
python3 get_moleculenet_embeddings.py --dataset_path=data/finetuning_datasets/classification --model_file=data/pretrained_models/modelO --heterodata_path=data/knowledge_graph/kg_heterodata.pt --dmgi_model=data/pretrained_models/kg_dmgi_model.pt
```

This command will generate embeddings for each dataset and save them as **{dataset_name}_modelO_embeddings.pkl** in the corresponding dataset directory. These files contain the following columns:
- **smiles:** SMILES notation of the molecule
- **label:** label of the molecule
- **selfies:** SELFIES notation of the molecule
- **sequence_embeddings:** embeddings of the SELFIES notation
- **text_embeddings:** embeddings of the textual description
- **unimol_embeddings:** embeddings of the 3D structure
- **kg_embeddings:** embeddings of the compounds in the knowledge graph

Generated embeddings can be used for training the multimodal classification model or generating predictions for new molecules using pre-trained classifiers.

<br/>

### 2. Training Multimodal Classification Model
For training the multimodal classification model for a dataset, you first need to generate embeddings for the dataset using the pre-trained models (please refer to the section "Generating Multimodal Embeddings Using Pre-trained Models"). After generating embeddings, you can train the multimodal classification model using the following command:

```
python3 train_classification_model.py --model=data/pretrained_models/modelO --tokenizer=data/RobertaFastTokenizer --dataset=data/finetuning_datasets/classification/bbbp/bbbp_modelO_embeddings.pkl --save_to=data/finetuned_models/bbbp --target_column_id=1 --use_scaffold=1 --train_batch_size=16 --validation_batch_size=8 --num_epochs=25 --lr=5e-5 --wd=0
```

TODO: update hyperparameters

This command will train the multimodal classification model using the embeddings generated for the dataset. The trained model will be saved in the specified directory. The model can be used for generating predictions for new molecules.

<br/>

## Generating Predictions for New Molecules
For generating predictions for a new dataset, you first need to generate embeddings for the dataset using the pre-trained models (please refer to the section "Generating Multimodal Embeddings Using Pre-trained Models"). After generating embeddings, you can use the following command to generate predictions for the molecules in the dataset:

```
python3 prediction.py --dataset=bbbp_mock --tokenizer_name=data/RobertaFastTokenizer --pred_set=data/finetuning_datasets/classification/bbbp_mock/bbbp_mock_modelO_embeddings.pkl --training_args=data/finetuned_models/bbbp/training_args.bin --model_name=data/finetuned_models/bbbp
```
This command will generate predictions for the molecules in the specified dataset using the pre-trained classifier. The predictions will be saved in the specified directory. 

Example dataset for generating predictions is provided in the **/data/finetuning_datasets/classification/bbbp_mock/** directory. Resulting predictions for this dataset can be found in the **/data/predictions/** directory.
