%this script generates all of the data needed for the manuscript
%GAGA for Compressed Sensing

% when uisng this file independently add the path to GAGA
%addpath "your path"/GAGA/gaga

gpuNumber = 0;

delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
delta_list=sort(delta_list);

rho_list=linspace(0,1,50);

n_list=[2^10 2^12 2^14 2^16 2^18 2^20]; 

%testing times increase significantly with n.
%the values used to generate the tables in 
%GAGA for Compressed Sensing take multiple days.
%those values are
% n_max_gen=2^14;
%the below values are selected for abridged testing.
n_max_gen=2^10;  % reduction for representative data

m_minimum=100;

maxiter_niht=5000; 
maxiter_htp=300;

tol=10^(-4);
alpha=0.25;

tests_per_k=10;  

supportFlag_list = [0];


tic

for sf=1:length(supportFlag_list)
  suppFlag=supportFlag_list(sf);

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

          % first test NIHT
          options = gagaOptions('vecDistribution','binary','alpha',alpha,'matrixEnsemble','gaussian','tol',tol*m/n,'supportFlag',suppFlag,'maxiter',maxiter_niht,'timing','on','gpuNumber',gpuNumber);
        
          if n<=n_max_gen
            success=0;
            for p=1:tests_per_k
              [en ct it supset rate xout] = gaga_cs('NIHT', 'gen', k, m, n, options);
              success = success+ ( en(end) < (tol*10) );
            end
            if success==0
              break;
            end
          end

          % second test HTP
          options = gagaOptions('vecDistribution','binary','alpha',alpha,'matrixEnsemble','gaussian','tol',tol*m/n,'supportFlag',suppFlag,'maxiter',maxiter_htp,'timing','on','gpuNumber',gpuNumber);
      
          if n<=n_max_gen
            success=0;
            for p=1:tests_per_k
              [en ct it supset rate xout] = gaga_cs('HTP', 'gen', k, m, n, options);
              success = success+ ( en(end) < (tol*10) );
            end
            if success==0
              break;
            end
          end
  
        end %ends k_list (which loops over rho_list)
      end %ends m_minimum
      display(sprintf('gen: m = %d for n = %d finished after %f seconds',m,n,toc))
    end %ends delta_list
  end %ends n_list
end % ends supportFlag_list loop

