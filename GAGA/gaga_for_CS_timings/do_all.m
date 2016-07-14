% this script generates and processes all data for the manuscript
% GAGA for Compressed Sensing

% Under the assumption this is run from ...../GAGA/gaga_for_CS_timings
% this adds the paths to gaga and gaga_matlab
tmp = pwd;
GAGApath = tmp(1:end-20);
addpath([GAGApath '/gaga']);
addpath([GAGApath '/gaga_matlab']);

% the data presented in the manuscript takes a substantial 
% time to generate.  the below generate_..._timing_data 
% scripts generate a subset of the date by limiting the 
% large problem size, n.  the values used for the manuscript 
% are also listed.  

generate_gpu_dct_timing_data
generate_gpu_smv_timing_data
generate_gpu_gen_timing_data

generate_matlab_NIHT_timing_data
generate_matlab_HTP_timing_data

process_timing_data_all

make_all_tables

make_all_plots

calculate_total_time

!pdflatex GAGA_table_and_plots.tex
!pdflatex GAGA_table_and_plots.tex

% the final command above will automatically 
% latex the file GAGA_table_and_plots.tex 
% which generates a file with the 
% tables and plots as formated in the 
% GAGA for Compressed Sensing manuscript

