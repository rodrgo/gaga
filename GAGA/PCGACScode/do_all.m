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
tables = 1;

% set algorithms to be tested
alg_list=cell(3,1);
alg_list{1}='NIHT';
alg_list{2}='HTP';
alg_list{3}='CSMPSP';

% set matrix ensembles to be tested
matens_list=cell(3,1);
matens_list{1}='dct';
matens_list{2}='smv';
matens_list{3}='gen';

% save the current path
CurrentDir = pwd;
addpath(CurrentDir)
addpath([CurrentDir '/shared'])

% set the path to GAGA
GAGApath = CurrentDir(1:end-11);
addpath([GAGApath '/gaga']);

% data for problem class (Mat,B)
% i.e. all matrix ensembles with sparse binary vectors
if binary_nonoise
cd([CurrentDir '/code'])
generate_data_timings(alg_list,matens_list);
generate_data_transition(alg_list,matens_list);
process_all_data(alg_list,matens_list);
make_all_plots
%make_error_separation_plot

display(sprintf('Completed do_all tasks for noiseless setting after %0.1f seconds.',toc(doalltime)));
end
% data for problem classes (Mat,vec_\epsilon)
% i.e. all matrix ensembles with vectors from 
% binary, uniform, normal distributions and 
% possible including additive noise.
if noise_and_altvecdistr
cd([CurrentDir '/code_noise'])
generate_data_timings_noise(alg_list,matens_list);
generate_data_transition_noise(alg_list,matens_list);
process_all_data_noise(alg_list,matens_list);
make_all_plots_noise


display(sprintf('Completed do_all tasks for noise and alternate vector distributions after %0.1f seconds.',toc(doalltime)));
end
% tables (this uses the actual values of (k,m,n) in the paper and takes approximately three days on a C2070)
% Here only the two smallest values of m are used for representative data with only 100 tests per k.
% The full data set is easily generated with minor changes to generate_data_table_noise.m.
if tables
cd([CurrentDir '/code_table'])
generate_data_table_noise
process_table_data
make_PCGACS_tables


display(sprintf('Completed do_all tasks for tables after %0.1f seconds.',toc(doalltime)));
end


