%turn on (1) or off (0) ensembles to be processed.
dct = 1;
smv = 1;
gen = 1;

% set the values of n for each matrix ensemble
n_list_dct=[2^17];
n_list_smv=[2^17];
n_list_gen=[2^13];
n_list_dct=[2^10];   % reduction for representative data
n_list_smv=[2^10];   % reduction for representative data
n_list_gen=[2^10];   % reduction for representative data

% set the algorithms
%{
alg_list=cell(5,1);
alg_list{1}='CGIHTrestarted';
alg_list{2}='CGIHT';
alg_list{3}='NIHT';
alg_list{4}='FIHT';
alg_list{5}='CSMPSP';
%alg_list{6}='HTP';
%}
alg_list=cell(3,1);
alg_list{1}='NIHT';
alg_list{2}='HTP';
alg_list{3}='CSMPSP';


nonzeros_list=[7];

noise_list = [0 0.1]
vec_list=[0 1 2];


for i=1:length(vec_list)
  for j=1:length(noise_list)

    if dct
      for q=1:length(n_list_dct)
        make_best_alg_plot_noise('dct',n_list_dct(q),0,noise_list(j),vec_list(i),alg_list,1);
      end
    end

    if smv
      for q=1:length(n_list_smv)
        for p=1:length(nonzeros_list)
          make_best_alg_plot_noise('smv',n_list_smv(q),nonzeros_list(p),noise_list(j),vec_list(i),alg_list,1);
        end
      end
    end

    if gen
      for q=1:length(n_list_gen)
        make_best_alg_plot_noise('gen',n_list_gen(q),0,noise_list(j),vec_list(i),alg_list,1);
      end
    end

  end  % ends noise_list loop
end  % ends vec_list loop


