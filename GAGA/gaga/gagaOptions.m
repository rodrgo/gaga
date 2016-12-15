function y = gagaOptions(varargin)
% This function takes pairs of option arguments for gaga, creates 
% a cell array of options which can then be passed to gaga as its
% final argument.
% Example usage: options = gagaOptions('tol',0.0001,'maxiter',400);
% [en ct it supset rate xout] = gaga_cs('NIHT','gen',2^4,2^8,2^9,options);
%
% Admissible options and their values: 
% tol: specifying convergence rate stopping conditions, positive float, default 0.001.
% maxiter: maximum number of iterations, positive integer, default 300 for HTP and
%          CSMPSP, 5000 for other algorithms.
% vecDistribution: distribution of the nonzeros in the sparse
%                  vector for the test problem instance, string
%                  options: 'binary' (default) for random plus and
%                  minus 1, 'uniform' for uniform from [0,1], and
%                  'gaussian' for normal N(0,1).  
% matrixEnsemble: distribution of the nonzeros in the measurement
%                 matrix for the test problem, string options:
%                 'gaussian' (default for gen, only valid for gen)
%                 for normal N(0,1), 'binary' (default for smv) for
%                 random plus and minus 1, and 'ones' (only valid
%                 for smv) for all ones. 
% seed: seed for random number generator, unsigned int, default clock().
% numBins: number of bins to use for order statistics, positive
%          integer, default to max(n/20,1000).
% threadsPerBlockn: number of threads per block for kernels acting
%                   on n vectors, positive integer, default to
%                   min(n, max_threads_per_block). 
% threadsPerBlockm: number of threads per block for kernels acting
%                   on m vectors, positive integer, default to
%                   min(m, max_threads_per_block). 
% threadsPerBlocknp: number of threads per block for kernels acting
%                    on vectors of length n*p, positive integer,
%                    default to min(nz, max_threads_per_block). 
% threadsPerBlockBin: number of threads per block for kernels
%                     acting on vectors of length numBins, positive
%                     integer, default to min(num_bins, max_threads_per_block).
% convRateNum: number of the last iterations to use when
%              calculating average convergence rate, positive
%              integer, default 16. 
% kFixed: flag to force the k used in the problem generate to be
%         that specified, string options: 'off' (default) and 'on'. 
% noise: level of noise as a fraction of the \|Ax\|_2, non-negative
%        float, default to 0.
% gpuNumber: which gpu to run the code on, non-negative integer, default to 0.
% timing: indicates that times per iteration should be recorded,
% string options: 'off' (default) and 'on'.
% alpha: specifying fraction of k used in early support set
%        identification steps, float between (0,1), default to 0.25.
%        (only valid with 'timing' set to 'on'.)
% supportFlag: method by which the support set is identified,
%              integer options: 0 (default) for dynamic binning
%              where binning is conducted only when the support set
%              could have changed, 1 for binning at every
%              iteration, 2 using thrust::sort to find the largest
%              entries when the support set could have changed, and
%              3 using thrust::sort at every iteration.
%              (only valid with 'timing' set to 'on'.)


  l=length(varargin);

  if mod(l,2)==1
    error('options must be submitted in pairs')
  elseif l~=0
    y=cell(l/2,2);
    for j=1:l/2
      y{j,1}=varargin{1+(j-1)*2};
      if isnumeric(varargin{2+(j-1)*2})
        %tmp=int32(varargin{2+(j-1)*2});
        %if abs(double(tmp)-varargin{2+(j-1)*2})==0
        %  y{j,2}=tmp;
        if ( strcmp(y{j,1}, 'debug_mode')==1 || strcmp(y{j,1}, 'l0_thresh')==1 || strcmp(y{j,1}, 'num_band_levels')==1 || strcmp(y{j,1},'maxiter')==1 || strcmp(y{j,1},'gpuNumber')==1 || strcmp(y{j,1},'supportFlag')==1 || strcmp(y{j,1},'convRateNum')==1 || strcmp(y{j,1},'numBins')==1 )
          y{j,2}=int32(varargin{2+(j-1)*2});
        elseif ( strcmp(y{j,1},'threadsPerBlockn')==1 || strcmp(y{j,1},'threadsPerBlockm')==1 || strcmp(y{j,1},'threadsPerBlocknp')==1 || strcmp(y{j,1},'threadsPerBlockBin')==1 )
          y{j,2}=int32(varargin{2+(j-1)*2});
        elseif ( strcmp(y{j,1},'seed')==1 )
          y{j,2}=uint32(varargin{2+(j-1)*2});
        else
          y{j,2}=single(varargin{2+(j-1)*2});
        end
      else
        y{j,2}=varargin{2+(j-1)*2};
      end
    end
  else %l==0
    % Set defaults
    y=cell(1);
    y{1}='default';
  end
  
  % if gpuNumber is passed as an option then 
  % it must come before thread options
  % so we put gpuNumber to be first
  if (l~=0 && mod(l,2)==0)
    for j=1:l/2
      if strcmpi(y{j,1},'gpuNumber')==1
        ind=j;
        
        tmp1=y{1,1}; tmp2=y{1,2};
        y{1,1}=y{ind,1}; y{1,2}=y{ind,2};
        y{ind,1}=tmp1; y{ind,2}=tmp2;
        
        break
      end
    end
  end
  
  % if numBins is passed as an option then 
  % it must come before threadsPerBlockBin
  % so we put threadsPerBlockBin to be last
  if (l~=0 && mod(l,2)==0)
    for j=1:l/2
      if strcmpi(y{j,1},'threadsPerBlockBin')==1
        ind=j;
        
        tmp1=y{l/2,1}; tmp2=y{l/2,2};
        y{l/2,1}=y{ind,1}; y{l/2,2}=y{ind,2};
        y{ind,1}=tmp1; y{ind,2}=tmp2;
        
        break
      end
    end
  end

  
