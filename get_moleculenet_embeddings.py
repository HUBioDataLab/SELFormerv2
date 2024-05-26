import os
from time import time
from fnmatch import fnmatch

import pandas as pd
from pandarallel import pandarallel
import to_selfies
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AutoTokenizer, AutoModel
from dmgi_model import load_heterodata, load_dmgi_model

import argparse

import numpy as np
from unimol_tools import UniMolRepr

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, metavar="/path/to/dataset/", help="Path of the input MoleculeNet datasets.")
parser.add_argument("--model_file", required=True, metavar="<str>", type=str, help="Name of the pretrained model.")
parser.add_argument("--heterodata_path", required=True, metavar="/path/to/heterodata/", help="Path of the input heterodata.")
parser.add_argument("--dmgi_model", required=True, metavar="/path/to/dmgi_model/", help="Path of the input dmgi_model.")

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_file = args.model_file # path of the pre-trained model
config = RobertaConfig.from_pretrained(model_file)
config.output_hidden_states = True
tokenizer = RobertaTokenizer.from_pretrained("./data/RobertaFastTokenizer")
model = RobertaModel.from_pretrained(model_file, config=config)

scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
unimol_model = UniMolRepr(data_type='molecule', remove_hs=False)

heterodata = load_heterodata(args.heterodata_path)               
dmgi_model = load_dmgi_model(args.dmgi_model, heterodata)

def generate_moleculenet_selfies(dataset_file):
    """
    Generates SELFIES for a given dataset and saves it to a file.
    :param dataset_file: path to the dataset file
    """

    dataset_name = dataset_file.split("/")[-1].split(".")[0]
    
    print(f'Generating SELFIES for {dataset_name}')

    if dataset_name == 'bace':
        smiles_column = 'mol'
    else:
        smiles_column = 'smiles'

    # read dataset
    dataset_df = pd.read_csv(os.path.join(dataset_file))
    dataset_df["selfies"] = dataset_df[smiles_column] # creating a new column "selfies" that is a copy of smiles_column

    # generate selfies
    pandarallel.initialize()
    dataset_df.selfies = dataset_df.selfies.parallel_apply(to_selfies.to_selfies)

    dataset_df.drop(dataset_df[dataset_df[smiles_column] == dataset_df.selfies].index, inplace=True)
    dataset_df.drop(columns=[smiles_column], inplace=True)
    out_name = dataset_name + "_selfies.csv"

    # save selfies to file
    path = os.path.dirname(dataset_file)

    dataset_df.to_csv(os.path.join(path, out_name), index=False)
    print(f'Saved to {os.path.join(path, out_name)}')


def get_sequence_embeddings(selfies, tokenizer, model):

    torch.set_num_threads(1)
    token = torch.tensor([tokenizer.encode(selfies, add_special_tokens=True, max_length=512, padding=True, truncation=True)])
    output = model(token)

    sequence_out = output[0]
    return torch.mean(sequence_out[0], dim=0).tolist()

def get_text_embeddings(text, tokenizer, model):
    
    torch.set_num_threads(1)

    if type(text) == str:
        token = torch.tensor([tokenizer.encode(text, add_special_tokens=True, max_length=512, padding=True, truncation=True)])
        output = model(token)
        text_out = output[0][0].mean(dim=0)
    else:
        text_out = torch.zeros(768)
        
    return text_out.tolist()

def get_unimol_embeddings(smiles, model):

    unimol_repr = model.get_repr(smiles, return_atomic_reprs=True) # UniMolRepr model
    
    # CLS token repr
    print(np.array(unimol_repr['cls_repr']).shape)
    
    # atomic level repr, align with rdkit mol.GetAtoms()
    print(np.array(unimol_repr['atomic_reprs']).shape)

def get_kg_embeddings(chembl_id, heterodata, dmgi_model):
    
    node_idx = heterodata['Compound'].id_mapping[chembl_id] if chembl_id in heterodata['Compound'].id_mapping else None
    if node_idx:    
        output = dmgi_model.Z[node_idx]
    else:
        output = torch.zeros(64)
        
    return output.tolist()

def generate_embeddings(model_file, heterodata, dmgi_model, args):
    root = args.dataset_path
    model_name = model_file.split("/")[-1]

    prepare_data_pattern = "*.csv"

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, prepare_data_pattern) and not any(substring in name for substring in ['selfies', 'embeddings', 'results']):
                dataset_file = os.path.join(path, name)
                dataset_name = dataset_file.split("/")[-1].split(".")[0]
                
                print(f'-------Processing {dataset_name}-------')
                
                generate_moleculenet_selfies(dataset_file)

                selfies_file = os.path.join(path, name.split(".")[0] + "_selfies.csv")

                print(f'Generating embeddings for {dataset_name}')
                t0 = time()

                dataset_df = pd.read_csv(selfies_file)
                pandarallel.initialize(nb_workers=10, progress_bar=True) # number of threads
                
                # selformer embeddings
                print(f'\n\nGenerating SELFormer embeddings using pre-trained model {model_name}')
                dataset_df["sequence_embeddings"] = dataset_df.selfies.parallel_apply(get_sequence_embeddings, args=(tokenizer, model))

                # scibert embeddings
                print(f'\n\nGenerating SciBERT embeddings')
                dataset_df["text_embeddings"] = dataset_df.description.parallel_apply(get_text_embeddings, args=(scibert_tokenizer, scibert_model))
                
                print(f'\n\nGenerating UniMol embeddings')
                dataset_df["unimol_embeddings"] = dataset_df.selfies.parallel_apply(get_unimol_embeddings, args=(unimol_model,))
                
                print(f'\n\nGenerating KG embeddings')
                dataset_df["kg_embeddings"] = dataset_df.chembl_id.parallel_apply(get_kg_embeddings, args=(heterodata, dmgi_model))
                
                dataset_df.drop(columns=["selfies", "description", "chembl_id"], inplace=True) # not interested in selfies data anymore, only class and the embedding
                file_name = f"{dataset_name}_{model_name}_embeddings.csv"

                # save embeddings to file
                path = os.path.dirname(selfies_file)
                dataset_df.to_csv(os.path.join(path, file_name), index=False)
                t1 = time()

                print(f'Finished in {round((t1-t0) / 60, 2)} mins')
                print(f'Saved to {os.path.join(path, file_name)}\n')


generate_embeddings(model_file, heterodata, dmgi_model, args)