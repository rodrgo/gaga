function y = logit_model(rho,b)
  b0=b(1);
  b1=b(2);
  y = 1./(1 + exp( -b0*(1 - b1 * rho) ));
