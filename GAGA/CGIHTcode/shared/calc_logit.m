function b = calc_logit(k,m,success,num_tests)
  
  options = optimset('TolFun',10^(-6),'MaxFunEvals',5000,'display','off');  %for fminunc
  
  %success is the fraction of times successful,
  %num_tests is the number of tests for that k,m pair
  
  rho=k./m;
  m=m(1);
  
  %have the initial approximation be a step function from 1 to 0
  %happening at one of the values of rho.
  
  err0=zeros(size(rho));
  for j=1:length(rho)
    err0(j)=logit_model_error(rho,success,num_tests,[4*m/rho(j) 1/rho(j)]);
  end
  
  ind=find(err0==min(err0));
  if length(ind)==1
    b1_tmp=1/rho(ind);
  else
    r=sum(rho(ind))/length(ind);
    b1_tmp=1/r;
  end
  
  r_tmp=1/b1_tmp;
  

  %the slope is between -m (change of one over 1/m width of rho)
  %and -1/(max(rho))
  
  slope=linspace(-m,-1/max(rho),200);
  b0_list=-4*slope*r_tmp;
  
  err=zeros(size(b0_list));
  b_list=zeros(length(b0_list),2);
  
  
  for j=1:length(b0_list)
    
    warning off
    b_tmp=[b0_list(j) b1_tmp];
    b_ans=fmincon(@(b) logit_model_error(rho,success,num_tests,b),b_tmp,[],[],[],[],[min(b0_list) 1],[max(b0_list) m],[],options);

    err(j)=logit_model_error(rho,success,num_tests,b_ans);
    b_list(j,:)=b_ans;
    
  end
  
  warning on
  
  
  ind=find(err==min(err));
  b=b_list(ind(ceil(length(ind)/2)),:);
  
