# ReplicateInstantNeuralMaterial
This is a unofficial replication of *Towards Comprehensive Neural Materials: Dynamic Structure-Preserving Synthesis with Accurate Silhouette at Instant Inference Speed*. The code of falcor implementation in this repo is adapted from git@github.com:Starry316/InstantNeuralMaterial.git.

## Components
This project contains two parts: 
1. training a quantised MLP model for a BTF material and export bundle
2. utilising exported feature bundle to render a neural material
## Training
###	Environment
Python3.8
btf-extractor
numpy
pytorch
### Run
1. put btf material under dataset folder 
2. run : python train.py --btf dataset/*.btf --epochs 300 --batch 2560000 --samples 25600000 --val_n 10 --reuse_per_frame 256 --cache_size 12 --qat --qat_calib_steps 200 --qat_freeze_after 100 --out runs/folder_name  --export_falcor_dir runs/folder_name/export_qtp --falcor_name name --accum_steps 1

## Rendering
### Build Falcor
The file is ready to build, more information in README.md under * InstantNeuralMaterial_copy * folder.

The implementation was tested on RedHat 9
### Run
Once built
1. place InstantNeuralMaterial_copy/neural_materials into Media directory
2. run .../.../Mogwai -s InstantNeuralMaterial_copy/bunny_neural_inference.py

## Note
Despite the rendering process can be ran, the implementation is incomplete. Dynamic synthesis doesn't work for now.
