

addpath /home/blanchard/GAGA/CGIHTcode/shared

  alg_list=cell(7,1);
  alg_list{1}='CGIHT';
  alg_list{2}='CGIHTrestarted';
  alg_list{3}='CGIHTprojected';
  alg_list{4}='NIHT';
  alg_list{5}='FIHT';
  alg_list{6}='CSMPSP';
  alg_list{7}='HTP';


  ens_list=cell(3,1);
  ens_list{1} = 'dct';
  ens_list{2} = 'smv';
  ens_list{3} = 'gen';
  
  process_all_data_noise(alg_list,ens_list);
