%this script generates all of the data needed for the manuscript
%GAGA for Compressed Sensing

% when uisng this file independently add the path to GAGA
%addpath "your path"/GAGA/gaga

gpuNumber = 0;

%need data for NIHT with all supp_flags in the list, and HTP for
%supp_flag = 0.


delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
delta_list=sort(delta_list);

rho_list=linspace(0,1,50);

n_list=[2^10 2^12 2^14 2^16 2^18 2^20]; 

%testing times increase significantly with n.
%the values used to generate the tables in 
%GAGA for Compressed Sensing take multiple days.
%those values are
% n_max_dct=2^20;
%the below values are selected for abridged testing.
n_max_dct=2^12;

m_minimum=100;

sorting_flag_list = [0 1 2 3];
% sorting_flag = 0 is for adaptively selecting the support size
%              = 1 is for using full binning at each step
%              = 2 is for using thrust::sort to find the kth largest magnitude with dynamic support detection
%              = 3 is for using thrust::sort at each step


maxiter_niht=5000; 
maxiter_htp=300;

tol=10^(-4);
alpha=0.25;

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
      for ll=1:length(k_list)
        k=k_list(ll);
      
        %for NIHT dct we test each of the sorting set flags
        for mm=1:length(sorting_flag_list)
          supp_flag=sorting_flag_list(mm);
      
          % first test NIHT
          options = gagaOptions('vecDistribution','binary','alpha',alpha,'tol',tol*m/n,'supportFlag',supp_flag,'maxiter',maxiter_niht,'timing','on','gpuNumber',gpuNumber);
          
          if n<=n_max_dct
            success=0;
            for p=1:tests_per_k
              [en ct it supset rate xout] = gaga_cs('NIHT', 'dct', k, m, n, options);
              success = success+ ( en(end) < (tol*10) );
            end
            if success==0
              break;
            end
          end
          
        end %end the sorting flag loop for NIHT dct
        
        %now test HTP, but only for support set flags = 0
      
        % second test HTP
        options = gagaOptions('vecDistribution','binary','alpha',alpha,'tol',tol*m/n,'supportFlag',0,'maxiter',maxiter_htp,'timing','on','gpuNumber',gpuNumber);

        if n<=n_max_dct
          success=0;
          for p=1:tests_per_k
            [en ct it supset rate xout] = gaga_cs('HTP', 'dct', k, m, n, options);
            success = success+ ( en(end) < (tol*10) );
          end
          if success==0
            break;
          end
        end
        
      end %ends k_list (which loops over rho_list)
    end %ends m_minimum
    display(sprintf('dct: m = %d for n = %d finished after %f seconds',m,n,toc))
  end %ends delta_list
end %ends n_list


