%this script generates all of the data needed for the manuscript
%GAGA for Compressed Sensing

% when uisng this file independently add the path to GAGA
%addpath "your path"/GAGA/gaga_matlab

%need data for NIHT with all supp_flags in the list, and HTP for
%supp_flag = 0.


delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
delta_list=sort(delta_list);

rho_list=linspace(0,1,50);

%decrease the delta and rho lists for faster testing
delta_list=delta_list(1:2:end);
rho_list=rho_list(1:2:end);

n_list=[2^10 2^12 2^14 2^16 2^18 2^20]; 


%testing times increase significantly with n.
%the values used to generate the tables in 
%GAGA for Compressed Sensing take multiple days.
%those values are
% n_max_dct=2^20;
% n_max_smv=2^20;
% n_max_gen=2^14;
%the below values are selected for abridged testing.
n_max_dct=2^12;
n_max_smv=2^12;
n_max_gen=2^10;

m_minimum=100;

nonZeroList=[4 7 13]; %smv has a specified number of nonzeros
                      %per column

sorting_flag_list = [0 1 2 3];
% sorting_flag = 0 is for adaptively selecting the support size
%              = 1 is for using full binning at each step
%              = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
%              = 3 is for using thrust::sort at each step


%use 5000 for NIHT and 100 or 300 for HTP
maxiter_niht=5000; 

tol=10^(-4);

tests_per_k=10;  


tic
for ii=1:length(n_list)
  n=n_list(ii);
  for jj=1:length(delta_list);
    m=ceil(n*delta_list(jj));
    if m>=m_minimum
      
      k_list=ceil(m*rho_list);
      k_list=max(k_list,1); k_list=min(k_list,m-1);
      k_list=intersect(k_list,k_list);
      
      %for NIHT dct we test each of the sorting set flags
      for mm=1:length(sorting_flag_list)
        supp_flag=sorting_flag_list(mm);
      
        options = gagaOptions('vecDistribution','binary','tol',tol*m/n,'supportFlag',supp_flag,'maxiter',maxiter_niht,'timing','on');

        %test the k from smallest until it fails all of the time
        for ll=1:length(k_list)
          k=k_list(ll);
        
          if n<=n_max_dct
            success=0;
            for p=1:tests_per_k
              global m;
              global n;
              [en ct it supset rate xout] = gaga_matlab_cs('NIHT', 'dct', k, m, n, options);
              success = success+ ( en(end) < (tol*10) );
            end
            if success==0
              break;
            end
          end
          
        end %end the k_list loop for NIHT dct
      end %end the sorting flag loop for NIHT dct
        
        
      %for gen and smv we test only support set flags = 0
          
      % second test gen
      options = gagaOptions('vecDistribution','binary','matrixEnsemble','gaussian','tol',tol*m/n,'supportFlag',0,'maxiter',maxiter_niht,'timing','on');
        
      %test the k from smallest until it fails all of the time
      for ll=1:length(k_list)
        k=k_list(ll);
      
        if n<=n_max_gen
          success=0;
          for p=1:tests_per_k
            global m;
            global n;
            [en ct it supset rate xout] = gaga_matlab_cs('NIHT', 'gen', k, m, n, options);
            success = success+ ( en(end) < (tol*10) );
          end
          if success==0
            break;
          end
        end
        
      end %end the k_list loop for NIHT gen
      
      
      % for smv we test each of the nonzero list
      % third test smv
      options = gagaOptions('vecDistribution','binary','matrixEnsemble','binary','tol',tol*m/n,'supportFlag',0,'maxiter',maxiter_niht,'timing','on');
        
      for qq=1:length(nonZeroList)
        %test the k from smallest until it fails all of the time
        for ll=1:length(k_list)
          k=k_list(ll);
        
          if n<=n_max_smv
            success=0;
            for p=1:tests_per_k
              global m;
              global n;
              [en ct it supset rate xout] = gaga_matlab_cs('NIHT', 'smv', k, m, n, nonZeroList(qq), options);
              success = success+ ( en(end) < (tol*10) );
            end
            if success==0
              break;
            end
          end
        end %end the k_list loop for NIHT smv
      end %end the nonZeroList loop for smv
      
    end %ends m_minimum
    display(sprintf('NIHT matlab: m = %d for n = %d finished after %f seconds',m,n,toc))
  end %ends delta_list
end %ends n_list


