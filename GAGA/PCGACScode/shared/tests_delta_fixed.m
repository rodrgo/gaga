function [k_min k_max] = tests_delta_fixed(alg,matens,k_min,k_max,m,n,nz,options,num_tests_per_k,num_k_to_test,noise_level,l_two_success)

%algorithms it knows are: 'ThresholdSD','ThresholdCG','IHT','NIHT','HTP','CSMPSP','CGIHT'

%using notation k \le m \le n with
%k, sparisty
%m, measurements
%n, signal length.

%vecDistribution determines how the nonzeros in the sparse vector are drawn.

%matrixEnsemble_smv determines how the nonzeros in the sparse
%matrix are drawn

%nonzeros is the number of nonzeros per column

if l_two_success
  success_tol = .001+2*noise_level;
  success_index = 2;
else
  success_tol = .001;
  success_index = 3;
end

threshold_success_tol = .01;

num_tests_find_transition=10;


%search for a k_min and k_max between which the phase transition
%occurs.  when searching for k_min and k_max have min_fixed and
%max_fixed set to zero.  when one of these is found, set the
%corresponding flag to be 1.
%k_min=1;  
%k_max=m-1;
rate_min=1;
rate_max=0;

%the following are used when finding k_min via bisection method
k_min_up=k_max;
k_min_mid=round((k_min+k_min_up)/2);


success_frac_boundary=0.95;
failure_frac_boundary=0.05;


%while ( (k_min_up - k_min) > ceil(2*sqrt(m)))
while ( (k_min_up - k_min) > ceil(sqrt(m)/10) )
  
  
  success=zeros(num_tests_find_transition,1);  
  for p=1:num_tests_find_transition
    if ( strcmp(alg,'ThresholdSD') || strcmp(alg,'ThresholdCG') )
      if strcmp(matens,'smv')
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_min_mid,m,n,nz,options);
      else
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_min_mid,m,n,options);
      end
      success(p) = ( en(success_index) < threshold_success_tol ); 
   else
      if strcmp(matens,'smv')
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_min_mid,m,n,nz,options);
      else
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_min_mid,m,n,options);
      end
      success(p) = ( en(success_index) < success_tol );
    end    
  end
  
  success_mid=sum(success)/length(success);
  if (success_mid > success_frac_boundary)
    k_min=k_min_mid;
  elseif (success_mid < success_frac_boundary)
    k_min_up=k_min_mid;
    if (success_mid < failure_frac_boundary) 
      k_max=k_min_mid;
    end
  end  
  k_min_mid=round((k_min+k_min_up)/2);
%  [k_min k_min_mid k_min_up]
%  success_mid
%  pause
  
end  


k_max_down=k_min;
k_max_mid=round((k_max+k_max_down)/2);

%while ( (k_max - k_max_down) > ceil(2*sqrt(m)))
while ( (k_max - k_max_down) > ceil(sqrt(m)/10) )
  
  
  success=zeros(num_tests_find_transition,1);  
  for p=1:num_tests_find_transition
    if ( strcmp(alg,'ThresholdSD') || strcmp(alg,'ThresholdCG_S_gen') )
      if strcmp(matens,'smv')
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_max_mid,m,n,nz,options);
      else
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_max_mid,m,n,options);
      end
      success(p) = ( en(success_index) < threshold_success_tol );
    else 
      if strcmp(matens,'smv')
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_max_mid,m,n,nz,options);
      else
        [en ct it supset rate xout]=gaga_cs(alg,matens,k_max_mid,m,n,options);
      end
      success(p) = ( en(success_index) < success_tol );
    end    
  end
  
  success_mid=sum(success)/length(success);
  if (success_mid < failure_frac_boundary)
    k_max=k_max_mid;
  elseif (success_mid > failure_frac_boundary)
    k_max_down=k_max_mid;
  end  
  k_max_mid=round((k_max+k_max_down)/2);
  
end  



k=round(linspace(k_min,k_max,num_k_to_test));
k=intersect(k,k);

for j=1:length(k)
  for p=1:num_tests_per_k
    if strcmp(matens,'smv')
      [en ct it supset rate xout]=gaga_cs(alg,matens,k(j),m,n,nz,options);
    else
      [en ct it supset rate xout]=gaga_cs(alg,matens,k(j),m,n,options);
    end
  end
end






