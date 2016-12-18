#!/bin/bash          
CUDAHOME="/usr/local/cuda-8.0"
GPUARCH="sm_30"
GAGA="/home/mendozasmith/src/robust_l0/GAGA_1_2_0/GAGA"
INCLUDEDIR="-I$(CUDAHOME)/include -I$(CUDAHOME)/NVIDIA_GPU_Computing_SDK/C/common/inc -I$(CUDAHOME)/NVIDIA_GPU_Computing_SDK/shared/inc -I/$(GAGA)"
INCLUDELIB="-L$(CUDAHOME)/lib64 -lcufft -lcurand -lcublas  -lcudart -Wl,-rpath,$(CUDAHOME)/lib64"
nvcc smv_debug.cu $(INCLUDEDIR) $(INCLUDELIB)

nvcc smv_debug.cu -g -G -I/usr/local/cuda-8.0/include -I/usr/local/cuda-8.0/NVIDIA_GPU_Computing_SDK/C/common/inc -I/usr/local/cuda-8.0/NVIDIA_GPU_Computing_SDK/shared/inc -I/home/mendozasmith/src/robust_l0/GAGA_1_2_0/GAGA -L/usr/local/cuda-8.0/lib64 -lcufft -lcurand -lcublas -lcudart -o smv_debug

