# STRANet
This repository includes:
- gen_file: Python version partition modes prediction 
- VVCSoftware_VTM-VTM-10.2-fast: VTM that loads the predicted partition modes from files and encodes with higher speed
- train_model: train prediction models
- VTM_libtorch: incorporate the prediction model into VTM using libtorch
## Usage
- Partition modes are first predicted by 'gen_file/gen_file.py', command:
```
python gen_file.py --model_path="./trained_models/" --seq_path="./yuv/"
```
- Use VVCSoftware_VTM-VTM-10.2-fast for fast intra coding. Before encoding, search for 'your_path' in source/Lib/EncoderLib/EncGOP.cpp to edit it to your folder path for the predicted partition modes.
## Requirements and Dependencies
- Pytorch 1.8.1+cu111
## Note
- gen_file/new_stf.py is based on https://github.com/Googolxx/STF
- The demo encodes 4 intra frames by default