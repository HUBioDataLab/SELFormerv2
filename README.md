# SELFormerv2

## get_moleculenet_embeddings usage
```
python3 get_moleculenet_embeddings.py --dataset_path=data/finetuning_datasets --model_file=data/pretrained_models/modelO --heterodata_path=data/knowledge_graph/kg_heterodata.pt --dmgi_model=data/pretrained_models/kg_dmgi_model.pt
```

To pre-train DMGI model or to use it for embedding, knowledge graph data (kg_heterodata.zip) file is needed to be downloaded from [here](https://drive.google.com/file/d/1u8kg7uzQ-q-osxIvrbJeAsFAbes1TmDF/view?usp=share_link) and extracted to data/knowledge_graph/ directory.


## train_classification_model usage

```
python3 train_classification_model.py --model=data/pretrained_models/modelO --tokenizer=data/RobertaFastTokenizer --dataset=data/finetuning_datasets/classification/bbbp/bbbp_modelO_unimol.pkl --save_to=data/finetuned_models/modelO_bbbp_classification --target_column_id=1 --use_scaffold=1 --train_batch_size=16 --validation_batch_size=8 --num_epochs=25 --lr=5e-5 --wd=0
```

example input data [here](https://drive.google.com/file/d/10qktsEChNcjMgpFvKepHz7sGn67xEbn1/view?usp=share_link)
