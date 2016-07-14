% this file processes all timings data and generates .mat files
% from which plots and tables are generated.

tic
%first we process the NIHT data
display(sprintf('Process timings begun.\n'));
process_NIHT_timings_data_gen
display(sprintf('NIHT for gen processed, %f total time.\n',toc));
process_NIHT_timings_data_dct
display(sprintf('NIHT for dct processed, %f total time.\n',toc));
process_NIHT_timings_data_smv
display(sprintf('NIHT for smv processed, %f total time.\n',toc));

process_HTP_timings_data_gen
display(sprintf('HTP for gen processed, %f total time.\n',toc));
process_HTP_timings_data_dct
display(sprintf('HTP for dct processed, %f total time.\n',toc));
process_HTP_timings_data_smv
display(sprintf('HTP for smv processed, %f total time.\n',toc));

process_matlab_NIHT_timings_data_gen
display(sprintf('NIHT matlab for gen processed, %f total time.\n',toc));
process_matlab_NIHT_timings_data_dct
display(sprintf('NIHT matlab for dct processed, %f total time.\n',toc));
process_matlab_NIHT_timings_data_smv
display(sprintf('NIHT matlab for smv processed, %f total time.\n',toc));

process_matlab_HTP_timings_data_gen
display(sprintf('HTP matlab for gen processed, %f total time.\n',toc));
process_matlab_HTP_timings_data_dct
display(sprintf('HTP matlab for dct processed, %f total time.\n',toc));
process_matlab_HTP_timings_data_smv
display(sprintf('HTP matlab for smv processed, %f total time.\n',toc));



