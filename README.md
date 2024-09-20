# STRANet
This repository includes:
- gen_dataset: generate dataset for training
- train_model: train prediction models
- gen_file: Python version partition modes prediction 
- VVCSoftware_VTM-VTM-10.2-fast: VTM that loads the predicted partition modes from files and encodes with higher speed
- VTM_libtorch: incorporate the prediction model into VTM using LibTorch
## Usage
- To generate the dataset, first use gen_dataset/flip.py to generate eight rotation types. Then use gen_dataset/VVCSoftware_VTM-VTM-10.2-dataset to get the ground truth partition modes. Finally, run gen_dataset_distrib_v2.py to get the probabilities for each partition mode.
- To train the model, run train_model/STRANet.py
- Partition modes are first predicted by 'gen_file/gen_file.py' (exmaples are availble at gen_file/C2), then 'gen_file/split_txt.py' can be used to split a txt into several txts, each for a frame. 
- Use VVCSoftware_VTM-VTM-10.2-fast for fast intra coding. Before encoding, search for 'your_path' in source/Lib/EncoderLib/EncGOP.cpp to edit it to your folder path for the predicted partition modes.
- See [VTM_libtorch/README.md](./VTM_libtorch/README.md) for the LibTorch version's Usage
## Requirements and Dependencies
- Pytorch 1.8.1+cu111
## Note
- gen_file/new_stf.py is based on https://github.com/Googolxx/STF
- The demo encodes 4 intra frames by default