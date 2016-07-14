
  alg_list=cell(6,1);
  alg_list{1}='CGIHT';
  alg_list{2}='CGIHTprojected';
  alg_list{3}='NIHT';
  alg_list{4}='FIHT';
  alg_list{5}='CSMPSP';
  alg_list{6}='HTP';

  generate_data_transition_noise(alg_list);
  generate_data_timings_noise(alg_list);
