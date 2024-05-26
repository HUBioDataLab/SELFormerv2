# SELFormerv2

## get_moleculenet_embeddings usage
```
python3 get_moleculenet_embeddings.py --dataset_path=data/finetuning_datasets --model_file=data/pretrained_models/modelO --heterodata_path=data/knowledge_graph/kg_heterodata.pt --dmgi_model=data/pretrained_models/kg_dmgi_model.pt
```

To pre-train DMGI model or to use it for embedding, knowledge graph data (kg_heterodata.zip) file is needed to be downloaded from [here](https://drive.google.com/file/d/1u8kg7uzQ-q-osxIvrbJeAsFAbes1TmDF/view?usp=share_link) and extracted to data/knowledge_graph/ directory.