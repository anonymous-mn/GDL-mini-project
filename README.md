# Surgical Fine-tuning for Graph Transformers

This codebase is based on the [GitHub repo](https://github.com/rampasek/GraphGPS/tree/main) of the paper "Recipe for a General, Powerful, Scalable Graph Transformer" (Rampasek et al., 2022). All modifications of the original codebase are marked with a comment containing the keyword "mjn". Modifications include new functions for transforming the datasets CIFAR10 and CLUSTER (to CIFAR10-Flip, Masking-CLUSTER and ML-CLUSTER), new config files for fine-tuning GPS models on the transformed datasets, code that allows freezing a subset of layers of a chosen model during fine-tuning, and an implementation of the Auto-RGN algorithm introduced in ["Surgical Fine-Tuning Improves Adaptation to Distribution Shifts"](https://arxiv.org/abs/2210.11466). For the GDL mini-project, this code was run on virtual machines ("computes") of the Microsoft Azure ML platform, in particular, on a Standard_D4s_v3 (4 cores, 16 GB RAM, 32 GB disk), and a Standard_F4s_v2 (4 cores, 8 GB RAM, 32 GB disk).

### Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```


### Fine-tuning a GraphGPS model
```bash
conda activate graphgps

# Pre-training GPS on the cluster dataset.
python main.py --cfg configs/GPS/cluster-GPS-ESLapPE.yaml  wandb.use False

# Fine-tuning GPS on one of the cluster fine-tuning tasks.
python main.py --cfg configs/GPS/cluster-FT-GPS-ESLapPE.yaml  wandb.use False
```

Note that freezing a subset of layers during fine-tuning can be achieved by modifying the variable "layers_to_freeze" in line 124 of main.py.

### This work is based on:

```bibtex
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Ramp\'{a}\v{s}ek and Mikhail Galkin and Vijay Prakash Dwivedi and Anh Tuan Luu and Guy Wolf and Dominique Beaini},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```