//This is for CPU encoding
#include "torch/torch.h"
#include "torch/script.h"
#include "FastPartition.h"
#include "UnitPartitioner.h"
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride){
      torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
      maxpool_options.stride(stride);
      return maxpool_options;
}

FastPartition::FastPartition()
{
}

void FastPartition::init_luma_feature_maps(int w, int h, int qp, int (*output_array)[550][970][9][9][4][7],int (*chroma_output_array)[40][70][5]){
  //setenv("CUDA_LAUNCH_BLCOK","1",1);
  torch::NoGradGuard no_grad_guard;
  torch::globalContext().setFlushDenormal(true);
  c10::InferenceMode guard;

  auto sT = std::chrono::system_clock::now();
  torch::jit::getExecutorMode()=false;
  
  for(int i=0;i<6;i++){
    res[i*2]=torch::jit::load("./Window_pt_models/"+std::to_string(i)+"/res-0.pt"); 
    //res[i*2].to(torch::kCUDA);
    res[i*2].eval();
    res[i*2+1]=torch::jit::load("./Window_pt_models/"+std::to_string(i)+"/res-1.pt"); 
    //res[i*2+1].to(torch::kCUDA);
    res[i*2+1].eval();
    subnet[i]=torch::jit::load("./Window_pt_models/"+std::to_string(i)+"/res-2.pt");
    //subnet[i].to(torch::kCUDA);
    subnet[i].eval();
  }
  auto eT = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(eT - sT).count();
  printf("load mode takes:%d microseconds\n",(int)duration);

  int batch_size_0=512;
  int batch_size_1=512;
  int batch_size_2=512;
  int batch_size_3=512;
  int batch_size_4=512;
  int batch_size_5=512;
  float threshold=0.15;
  
  //chroma
  torch::nn::MaxPool2d maxpool=torch::nn::MaxPool2d(maxpool_options(2,2));
  torch::Tensor y_down = maxpool(org_imageBatch);
  torch::Tensor cattensors = torch::cat({y_down,uv_imageBatch},1);
  //cattensors=cattensors.to(torch::kFloat32);
  int pos_list[4*(h/32)*(w/32)][3];//frame_num,h,w
  std::vector<torch::Tensor> input_list;
  int input_size=0;

  for(int frame_num=0;frame_num<4;frame_num++){
    for(int x=0;x<=h/2-32;x+=32){
      for(int y=0;y<=w/2-32;y+=32){
        pos_list[input_size][0]=frame_num;
        pos_list[input_size][1]=x;
        pos_list[input_size][2]=y;
        input_list.push_back(cattensors.slice(0,frame_num,frame_num+1).slice(2,x,x+32).slice(3,y,y+32));
        input_size++;
      }
    }
  }

  torch::TensorList tensorlist{input_list};
  cattensors = torch::cat(tensorlist).to(torch::kFloat32);

  for(int k=0;k<int(input_size/batch_size_5)+1;k++){
    if(k*batch_size_5==input_size)continue;
    int end_idx=(k+1)*batch_size_5;
    int batch_end_idx=batch_size_5;
    if(input_size<(k+1)*batch_size_5){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_5;
    }
    int start_idx=k*batch_size_5;
    input.push_back(cattensors.slice(0,k*batch_size_5,end_idx));//.to(torch::kCUDA));
    feature_tensor[0]=(torch::Tensor)(res[10].forward(input).toTensor());
    input.pop_back();
     
    input.push_back(feature_tensor[0]);
    feature_tensor[1]=(torch::Tensor)(res[11].forward(input).toTensor());//.to(torch::kCUDA));
    input.pop_back();
    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp).to(torch::kCUDA);

    input.push_back(feature_tensor[1]);
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[5].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
    
    
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      int frame=pos_list[batch_item+start_idx][0];
      int posh=pos_list[batch_item+start_idx][1];
      int posw=pos_list[batch_item+start_idx][2];
      for(int i=0;i<4;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold;
        chroma_output_array[frame][posh/32][posw/32][i]=split_list[i];
      }
      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      chroma_output_array[frame][posh/32][posw/32][4]=1;
    }
  }
  
  //32x32
  input_size=0;
  input_list=std::vector<torch::Tensor>();
  for(int frame_num=0;frame_num<4;frame_num++){
    for(int x=0;x<=h-32;x+=32){
      for(int y=0;y<=w-32;y+=32){
        pos_list[input_size][0]=frame_num;
        pos_list[input_size][1]=x;
        pos_list[input_size][2]=y;
        input_list.push_back(org_imageBatch.slice(0,frame_num,frame_num+1).slice(2,x,x+32).slice(3,y,y+32));
        input_size++;
      }
    }
  }
  
  class pos_item5{
    public:
    pos_item5(int _frame, int _posh, int _posw, int _cuh, int _cuw){
      frame=_frame; posh=_posh; posw=_posw; cuh=_cuh; cuw=_cuw;
    }
    int frame,posh,posw,cuh,cuw;
  };
  class pos_item3{
    public:
    pos_item3(int _frame, int _posh, int _posw){
      frame=_frame; posh=_posh; posw=_posw;
    }
    int frame,posh,posw;
  };
  class pos_item6{
    public:
    pos_item6(int _frame, int _posh, int _posw, int _cuh, int _cuw, int _depth){
      frame=_frame; posh=_posh; posw=_posw; cuh=_cuh; cuw=_cuw; depth=_depth;
    }
    int frame,posh,posw,cuh,cuw,depth;
  };
  std::vector<torch::Tensor> input_list_2, input_list_3, input_list_4, input_list_5;
  std::vector<pos_item5> pos_list_2, pos_list_5;
  std::vector<pos_item6> pos_list_4;
  std::vector<pos_item3> pos_list_3;
  std::vector<int> qp_list_2, qp_list_3, qp_list_4, qp_list_5;
  
  tensorlist=torch::TensorList{input_list};
  cattensors = torch::cat(tensorlist);
  for(int k=0;k<int(input_size/batch_size_0)+1;k++){
    if(k*batch_size_0==input_size)continue;
    int end_idx=(k+1)*batch_size_0;
    int batch_end_idx=batch_size_0;
    if(input_size<(k+1)*batch_size_0){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_0;
    }
    int start_idx=k*batch_size_0;
    
    input.push_back(cattensors.slice(0,k*batch_size_0,end_idx));//.to(torch::kCUDA));
    feature_tensor[0]=(torch::Tensor)(res[0].forward(input).toTensor());
    input.pop_back();

    input.push_back(feature_tensor[0]);
    feature_tensor[1]=(torch::Tensor)(res[1].forward(input).toTensor());
    input.pop_back();

    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp);//.to(torch::kCUDA);
    input.push_back(feature_tensor[1]);
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[0].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
   
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      int frame=pos_list[batch_item+start_idx][0];
      int posh=pos_list[batch_item+start_idx][1];
      int posw=pos_list[batch_item+start_idx][2];

      for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold;
        output_array[frame][posh/4][posw/4][8][8][0][i]=split_list[i];
      }
      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
        output_array[frame][posh/4][posw/4][8][8][0][torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      output_array[frame][posh/4][posw/4][8][8][0][6]=1;
      if(split_list[1]==1){
        input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw,posw+16));
        pos_list_3.push_back(pos_item3(frame,posh,posw));
        input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+16,posh+32).slice(3,posw,posw+16));
        pos_list_3.push_back(pos_item3(frame,posh+16,posw));
        input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw+16,posw+32));
        pos_list_3.push_back(pos_item3(frame,posh,posw+16));
        input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+16,posh+32).slice(3,posw+16,posw+32));
        pos_list_3.push_back(pos_item3(frame,posh+16,posw+16));
        qp_list_3.push_back(0);qp_list_3.push_back(0);qp_list_3.push_back(0);qp_list_3.push_back(0);
      }
      if(split_list[2]==1){
        input_list_2.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw,posw+32));
        pos_list_2.push_back(pos_item5(frame,posh,posw,16,32));
        input_list_2.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+16,posh+32).slice(3,posw,posw+32));
        pos_list_2.push_back(pos_item5(frame,posh+16,posw,16,32));
        qp_list_2.push_back(1);qp_list_2.push_back(1);
      }
      if(split_list[3]==1){
        input_list_2.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw,posw+16).transpose(3,2));
        pos_list_2.push_back(pos_item5(frame,posh,posw,32,16));
        input_list_2.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw+16,posw+32).transpose(3,2));
        pos_list_2.push_back(pos_item5(frame,posh,posw+16,32,16));
        qp_list_2.push_back(1);qp_list_2.push_back(1);
      }
      if(split_list[4]==1){
        input_list_2.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+8,posh+24).slice(3,posw,posw+32));
        pos_list_2.push_back(pos_item5(frame,posh+8,posw,16,32));
        qp_list_2.push_back(0);
        input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+8).slice(3,posw,posw+32));
        pos_list_4.push_back(pos_item6(frame,posh,posw,8,32,1));
        input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+24,posh+32).slice(3,posw,posw+32));
        pos_list_4.push_back(pos_item6(frame,posh+24,posw,8,32,1));
        qp_list_4.push_back(1);qp_list_4.push_back(1);
      }
      if(split_list[5]==1){
        input_list_2.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw+8,posw+24).transpose(3,2));
        pos_list_2.push_back(pos_item5(frame,posh,posw+8,32,16));
        qp_list_2.push_back(0);
        input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw,posw+8).transpose(3,2));
        pos_list_4.push_back(pos_item6(frame,posh,posw,32,8,1));
        input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw+24,posw+32).transpose(3,2));
        pos_list_4.push_back(pos_item6(frame,posh,posw+24,32,8,1));
        qp_list_4.push_back(1);qp_list_4.push_back(1);
      }
    }
  }
  //16x32
  input_size=input_list_2.size();
  tensorlist= torch::TensorList{input_list_2};
  cattensors = torch::cat(tensorlist);
  torch::Tensor qptensor = torch::from_blob(qp_list_2.data(), qp_list_2.size(), torch::dtype(torch::kInt32));//.unsqueeze(0);

  for(int k=0;k<int(input_size/batch_size_1)+1;k++){
    if(k*batch_size_1==input_size)continue;
    int end_idx=(k+1)*batch_size_1;
    int batch_end_idx=batch_size_1;
    if(input_size<(k+1)*batch_size_1){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_1;
    }
    int start_idx=k*batch_size_1;
    input.push_back(cattensors.slice(0,k*batch_size_1,end_idx));//.to(torch::kCUDA));
    feature_tensor[0]=(torch::Tensor)(res[4].forward(input).toTensor());
    input.pop_back();

    input.push_back(feature_tensor[0]);
    feature_tensor[1]=(torch::Tensor)(res[5].forward(input).toTensor());
    input.pop_back();
    
    torch::Tensor input_atten=torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*2;
    input_atten= (input_atten+qptensor.slice(0,start_idx,end_idx));//.to(torch::kCUDA);
    input.push_back(feature_tensor[1]);
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[2].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
    
    
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      int frame=pos_list_2[batch_item+start_idx].frame;
      int posh=pos_list_2[batch_item+start_idx].posh;
      int posw=pos_list_2[batch_item+start_idx].posw;
      int cuh=pos_list_2[batch_item+start_idx].cuh;
      int cuw=pos_list_2[batch_item+start_idx].cuw;
      for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold;
      }
      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][0]=split_list[0];
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][1]=split_list[1];
      if(cuh==32){
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][2]=split_list[3];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][3]=split_list[2];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][4]=split_list[5];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][5]=split_list[4];
      }
      else{
        for(int i=2;i<6;i++)output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][i]=split_list[i];
      }
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][0][6]=1;

      if(split_list[2]==1){
        if(cuh==16){
          input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+8).slice(3,posw,posw+32));
          pos_list_4.push_back(pos_item6(frame,posh,posw,8,32,2));
          input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+8,posh+16).slice(3,posw,posw+32));
          pos_list_4.push_back(pos_item6(frame,posh+8,posw,8,32,2));
          qp_list_4.push_back(1);qp_list_4.push_back(1);
        }
        else{
          input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw,posw+8).transpose(3,2));
          pos_list_4.push_back(pos_item6(frame,posh,posw,32,8,2));
          input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw+8,posw+16).transpose(3,2));
          pos_list_4.push_back(pos_item6(frame,posh,posw+8,32,8,2));
          qp_list_4.push_back(1);qp_list_4.push_back(1);
        }
      }
      if(split_list[3]==1){
        input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw,posw+16));
        pos_list_3.push_back(pos_item3(frame,posh,posw));
        qp_list_3.push_back(1);
        if(cuh==16){
          input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw+16,posw+32));
          pos_list_3.push_back(pos_item3(frame,posh,posw+16));
          qp_list_3.push_back(1);
        }
        else{
          input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+16,posh+32).slice(3,posw,posw+16));
          pos_list_3.push_back(pos_item3(frame,posh+16,posw));
          qp_list_3.push_back(1);
        }
      }
      if(split_list[4]==1){
        if(cuh==16){
          input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+4,posh+12).slice(3,posw,posw+32));
          pos_list_4.push_back(pos_item6(frame,posh+4,posw,8,32,2));
          qp_list_4.push_back(0);
        }
        else{
          input_list_4.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+32).slice(3,posw+4,posw+12).transpose(3,2));
          pos_list_4.push_back(pos_item6(frame,posh,posw+4,32,8,2));
          qp_list_4.push_back(0);
        }
      }
      if(split_list[5]==1){
        if(cuh==16){
          input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw+8,posw+24));
          pos_list_3.push_back(pos_item3(frame,posh,posw+8));
          qp_list_3.push_back(3);

          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw,posw+8).transpose(3,2));
          pos_list_5.push_back(pos_item5(frame,posh,posw,16,8));
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw+24,posw+32).transpose(3,2));
          pos_list_5.push_back(pos_item5(frame,posh,posw+24,16,8));
          qp_list_5.push_back(2);qp_list_5.push_back(2);
        }
        else{
          input_list_3.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+8,posh+24).slice(3,posw,posw+16));
          pos_list_3.push_back(pos_item3(frame,posh+8,posw));
          qp_list_3.push_back(2);

          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+8).slice(3,posw,posw+16));
          pos_list_5.push_back(pos_item5(frame,posh,posw,8,16));
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+24,posh+32).slice(3,posw,posw+16));
          pos_list_5.push_back(pos_item5(frame,posh+24,posw,8,16));
          qp_list_5.push_back(2);qp_list_5.push_back(2);
        }
      }
    }
  }
  //16x16模型
  input_size=input_list_3.size();
  tensorlist= torch::TensorList{input_list_3};
  cattensors = torch::cat(tensorlist);
  qptensor = torch::from_blob(qp_list_3.data(), qp_list_3.size(), torch::dtype(torch::kInt32));
  for(int k=0;k<int(input_size/batch_size_2)+1;k++){
    if(k*batch_size_2==input_size)continue;
    int end_idx=(k+1)*batch_size_2;
    int batch_end_idx=batch_size_2;
    if(input_size<(k+1)*batch_size_2){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_2;
    }
    int start_idx=k*batch_size_2;
    input.push_back(cattensors.slice(0,k*batch_size_2,end_idx));//.to(torch::kCUDA));
    feature_tensor[0]=(torch::Tensor)(res[2].forward(input).toTensor());
    input.pop_back();

    input.push_back(feature_tensor[0]);
    feature_tensor[1]=(torch::Tensor)(res[3].forward(input).toTensor());
    input.pop_back();

    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*4
                              +qptensor.slice(0,start_idx,end_idx));//.to(torch::kCUDA);
    input.push_back(feature_tensor[1]);
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[1].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
    
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      int frame=pos_list_3[batch_item+start_idx].frame;
      int posh=pos_list_3[batch_item+start_idx].posh;
      int posw=pos_list_3[batch_item+start_idx].posw;
      int cuh=16; int cuw=16;
      for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold;
      }
      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      int mode=qp_list_3[batch_item+start_idx];
      for(int i=0;i<6;i++)output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][i]=split_list[i];
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][6]=1;
      
      if(qp_list_3[batch_item+start_idx]!=0)continue;

      if(split_list[2]==1){
        input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+8).slice(3,posw,posw+16));
        pos_list_5.push_back(pos_item5(frame,posh,posw,8,16));
        input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+8,posh+16).slice(3,posw,posw+16));
        pos_list_5.push_back(pos_item5(frame,posh+8,posw,8,16));
        qp_list_5.push_back(2);qp_list_5.push_back(2);
      }
      if(split_list[3]==1){
        input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw,posw+8).transpose(3,2));
        pos_list_5.push_back(pos_item5(frame,posh,posw,16,8));
        input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw+8,posw+16).transpose(3,2));
        pos_list_5.push_back(pos_item5(frame,posh,posw+8,16,8));
        qp_list_5.push_back(2);qp_list_5.push_back(2);
      }
      if(split_list[4]==1){
        input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+4,posh+12).slice(3,posw,posw+16));
        pos_list_5.push_back(pos_item5(frame,posh+4,posw,8,16));
        qp_list_5.push_back(0);
      }
      if(split_list[5]==1){
        input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw+4,posw+12).transpose(3,2));
        pos_list_5.push_back(pos_item5(frame,posh,posw+4,16,8));
        qp_list_5.push_back(0);
      }
    }
  }

  //8x32
  input_size=input_list_4.size();
  tensorlist= torch::TensorList{input_list_4};
  cattensors = torch::cat(tensorlist);
  qptensor = torch::from_blob(qp_list_4.data(), qp_list_4.size(), torch::dtype(torch::kInt32));
  for(int k=0;k<int(input_size/batch_size_3)+1;k++){
    if(k*batch_size_3==input_size)continue;
    int end_idx=(k+1)*batch_size_3;
    int batch_end_idx=batch_size_3;
    if(input_size<(k+1)*batch_size_3){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_3;
    }
    int start_idx=k*batch_size_3;
    input.push_back(cattensors.slice(0,k*batch_size_3,end_idx));//.to(torch::kCUDA));
    feature_tensor[0]=(torch::Tensor)(res[6].forward(input).toTensor());
    input.pop_back();

    input.push_back(feature_tensor[0]);
    feature_tensor[1]=(torch::Tensor)(res[7].forward(input).toTensor());
    input.pop_back();

    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*2
                              +qptensor.slice(0,start_idx,end_idx));//.to(torch::kCUDA);
    input.push_back(feature_tensor[1]);
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[3].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
    
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      int frame=pos_list_4[batch_item+start_idx].frame;
      int posh=pos_list_4[batch_item+start_idx].posh;
      int posw=pos_list_4[batch_item+start_idx].posw;
      int cuh=pos_list_4[batch_item+start_idx].cuh;
      int cuw=pos_list_4[batch_item+start_idx].cuw;
      for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold;
      }
      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      
      int mode=qp_list_4[batch_item+start_idx];
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][0]=split_list[0];
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][1]=split_list[1];
      if(cuh==32){
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][2]=split_list[3];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][3]=split_list[2];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][4]=split_list[5];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][5]=split_list[4];
      }
      else{
        for(int i=2;i<6;i++)output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][i]=split_list[i];
      }
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][6]=1;

      if(pos_list_4[batch_item+start_idx].depth==2)continue; //终止条件

      if(split_list[3]==1){
        if(cuh==8){
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+8).slice(3,posw,posw+16));
          pos_list_5.push_back(pos_item5(frame,posh,posw,8,16));
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+8).slice(3,posw+16,posw+32));
          pos_list_5.push_back(pos_item5(frame,posh,posw+16,8,16));
          qp_list_5.push_back(2);qp_list_5.push_back(2);
        }
        else{
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+16).slice(3,posw,posw+8).transpose(3,2));
          pos_list_5.push_back(pos_item5(frame,posh,posw,16,8));
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+16,posh+32).slice(3,posw,posw+8).transpose(3,2));
          pos_list_5.push_back(pos_item5(frame,posh+16,posw,16,8));
          qp_list_5.push_back(2);qp_list_5.push_back(2);
        }
      }
      if(split_list[5]==1){
        if(cuh==8){
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh,posh+8).slice(3,posw+8,posw+24));
          pos_list_5.push_back(pos_item5(frame,posh,posw+8,8,16));
          qp_list_5.push_back(1);
        }
        else{
          input_list_5.push_back(org_imageBatch.slice(0,frame,frame+1).slice(2,posh+8,posh+24).slice(3,posw,posw+8).transpose(3,2));
          pos_list_5.push_back(pos_item5(frame,posh+8,posw,16,8));
          qp_list_5.push_back(1);
        }
      }
    }
  }

  //8x16
  input_size=input_list_5.size();
  tensorlist= torch::TensorList{input_list_5};
  cattensors = torch::cat(tensorlist);
  qptensor = torch::from_blob(qp_list_5.data(), qp_list_5.size(), torch::dtype(torch::kInt32));
  for(int k=0;k<int(input_size/batch_size_4)+1;k++){
    if(k*batch_size_4==input_size)continue;
    int end_idx=(k+1)*batch_size_4;
    int batch_end_idx=batch_size_4;
    if(input_size<(k+1)*batch_size_4){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_4;
    }
    int start_idx=k*batch_size_4;
    input.push_back(cattensors.slice(0,k*batch_size_4,end_idx));//.to(torch::kCUDA));
    feature_tensor[0]=(torch::Tensor)(res[8].forward(input).toTensor());
    input.pop_back();

    input.push_back(feature_tensor[0]);
    feature_tensor[1]=(torch::Tensor)(res[9].forward(input).toTensor());
    input.pop_back();

    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*3
                              +qptensor.slice(0,start_idx,end_idx));//.to(torch::kCUDA);
    input.push_back(feature_tensor[1]);
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[4].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
    
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      int frame=pos_list_5[batch_item+start_idx].frame;
      int posh=pos_list_5[batch_item+start_idx].posh;
      int posw=pos_list_5[batch_item+start_idx].posw;
      int cuh=pos_list_5[batch_item+start_idx].cuh;
      int cuw=pos_list_5[batch_item+start_idx].cuw;
      for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold;
      }

      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      
      int mode=qp_list_5[batch_item+start_idx];
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][0]=split_list[0];
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][1]=split_list[1];
      if(cuh==16){
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][2]=split_list[3];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][3]=split_list[2];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][4]=split_list[5];
        output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][5]=split_list[4];
      }
      else{
        for(int i=2;i<6;i++)output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][i]=split_list[i];
      }
      output_array[frame][posh/4][posw/4][cuh/4][cuw/4][mode][6]=1;
    }
  }
}
