# Usage
- Download Libtorch
- Edit the path for libtorch in CMakeLists.txt
- The demo encodes 4 intra frames by default
- Default input bit depth is 8 bit. If input is 10 bit, search for is_8bit and set it to false. 
- The demo use GPU for inference by default. Replace FastPartition.cpp with the FastPartition.cpp under this folder, then CPU will be used.
- The demo load