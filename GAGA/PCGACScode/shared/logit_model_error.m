function err = logit_model_error(rho,success,num_tests,b)
  b0=b(1);
  b1=b(2);
  y=logit_model(rho,b);
  err=(num_tests'*abs(success-y))/(max(1,num_tests'*success));

