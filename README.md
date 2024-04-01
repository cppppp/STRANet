# STRANet
This repository includes:
- Python version partition modes prediction 
- VTM that loads the predicted partition modes from files and encodes with higher speed
## Usage
- Partition modes are first predicted by 'gen_file/gen_file.py', command:
```
python gen_file.py --model_path="./trained_models/" --seq_path="./yuv/"
```
- Use VVCSoftware_VTM-VTM-10.2-fast for fast intra coding. Before encoding, search for 'your_path' in source/Lib/EncoderLib/EncGOP.cpp to edit it to your folder path for the predicted partition modes.
## Note
- gen_file/new_stf.py is based on https://github.com/Googolxx/STF
- More code will be released soon, including a libtorch version for partition modes prediction.