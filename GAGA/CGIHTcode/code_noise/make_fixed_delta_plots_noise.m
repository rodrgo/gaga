% make_fixed_delta_plots_noise.m


%n_gen=2^13;
%n_smv=2^17;
%n_dct=2^17;
n_gen=2^10;
n_smv=2^10;
n_dct=2^10;

alg_list=cell(7,1);
alg_list{1}='CGIHT';
alg_list{2}='CGIHTrestarted';
alg_list{3}='CGIHTprojected';
alg_list{4}='FIHT';
alg_list{5}='NIHT';
alg_list{6}='CSMPSP';
alg_list{7}='HTP';

ens_list=cell(3,1);
ens_list{1}='dct';
ens_list{2}='smv';
ens_list{3}='gen';


nonzeros_list=[7];
rowDistribution=1;

noise_list = [0 0.1 0.2];

 for pp=1:length(ens_list)
   matens = ens_list{pp};
       for nl=1:length(noise_list)
         if strcmp(ens_list{pp},'smv')
           for nz=1:length(nonzeros_list)
             nonzeros=nonzeros_list(nz);
             make_fixedmn_timing_plots_noise(alg_list, matens, n_smv, noise_list(nl), nonzeros)
           end  
         elseif strcmp(ens_list{pp},'dct')
           make_fixedmn_timing_plots_noise(alg_list, matens, n_dct, noise_list(nl), 0)
         else
           make_fixedmn_timing_plots_noise(alg_list, matens, n_gen, noise_list(nl), 0)
         end % ends if smv
       end % ends noise_list loop
 end % ends ens_list loop

 display('Completed fixedmn plots.')

