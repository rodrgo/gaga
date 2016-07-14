% do_all.m
%
% This script will generate all data for PCGACS,
% process the data, and create the tables and figures.
% To create representative data, the value of n is small.
% To complete all testsing, processing, and figure/table generation
% for the full range of n tested in PCGACS takes more than one month.
% The values of n are easily changed in each of the functions called below.

doalltime=tic;

% On (1) and Off (0) for each set of tasks
binary_nonoise = 1;
noise_and_altvecdistr = 1;

% set algorithms to be tested
alg_list=cell(7,1);
alg_list{1}='CGIHT';
alg_list{2}='CGIHTrestarted';
alg_list{3}='CGIHTprojected';
alg_list{4}='NIHT';
alg_list{5}='FIHT';
alg_list{6}='CSMPSP';
alg_list{7}='HTP';

% data for CGIHTrestarted is generated as CGIHT with the option restartFlag on.
alg_list_generate=cell(6,1);
alg_list_generate{1}=alg_list{1};
for j=2:6
  alg_list_generate{j}=alg_list{j+1};
end


% set matrix ensembles to be tested
matens_list=cell(3,1);
matens_list{1}='dct';
matens_list{2}='smv';
matens_list{3}='gen';

% save the current path
CurrentDir = pwd;
addpath(CurrentDir)
addpath([CurrentDir '/shared'])

% set path to gaga
GAGApath = CurrentDir(1:end-10);
addpath([GAGApath '/gaga'])

% data for problem class (Mat,B)
% i.e. all matrix ensembles with sparse binary vectors
if binary_nonoise
cd([CurrentDir '/code'])
generate_data_timings(alg_list_generate,matens_list);
generate_data_transition(alg_list_generate,matens_list);
process_all_data(alg_list,matens_list);
make_all_plots(alg_list,matens_list);
%make_error_separation_plot

display(sprintf('Completed do_all tasks for noiseless setting after %0.1f seconds.',toc(doalltime)));
end
% data for problem classes (Mat,B_\epsilon)
% i.e. all matrix ensembles with vectors from 
% binary including additive noise.
if noise_and_altvecdistr
cd([CurrentDir '/code_noise'])
generate_data_timings_noise(alg_list_generate,matens_list);
generate_data_transition_noise(alg_list_generate,matens_list);
process_all_data_noise(alg_list,matens_list);
make_all_plots_noise(alg_list,matens_list);


display(sprintf('Completed do_all tasks for noise and alternate vector distributions after %0.1f seconds.',toc(doalltime)));
end


display(sprintf('Completed do_all after %0.1f seconds.',toc(doalltime)));


