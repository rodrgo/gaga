% This is greedyheader.m which replicates greedyheader.cu.
% You must change the path in Location to the appropraite path for
% GAGA_matlab.

tmp = pwd;
ind=strfind(tmp,'/GAGA/');
Location=[tmp(1:ind) 'GAGA/gaga_matlab/'];

addpath([Location 'matrices'])
addpath([Location 'createProblem'])
addpath([Location 'kernels'])
addpath([Location 'functions'])
addpath([Location 'algorithms'])
addpath([Location 'algorithms_timings'])
addpath([Location 'results'])
