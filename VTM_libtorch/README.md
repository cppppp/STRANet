# Usage
- Download Libtorch (libtorch-shared-with-deps-1.13.0+cu116 is recommended.)
- Edit the path for libtorch in CMakeLists.txt.
- The demo encodes 4 intra frames by default.
- Default input bit depth is 8 bit. If input is 10 bit, search for is_8bit and set it to false. 
- The demo use GPU for inference by default. Replace FastPartition.cpp with the FastPartition.cpp which is under this folder, then CPU will be used.
- search for 'Window_pt_models' and edit its path. Replace 'Window_pt_models' with 'ResNet_pt_models' to change the model.
- The default configuration is C2. For the other three configurations, search for 'thres=' to edit the threshold. For C3 configuration, search for 'temp!=0 &&' and delete it.