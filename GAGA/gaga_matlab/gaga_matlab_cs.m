function [out1 out2 out3 out4 out5 out6] = gaga_matlab_cs(alg, ens, in3, ...
                                                  in4, in5, in6, ...
                                                  in7)

% Need to add help info...

if strcmp(ens,'gen')
  % [norms times iterations support convRate xout] = gaga_gen(algstring,k,m,n,options)
    if (nargin == 5)
      [out1 out2 out3 out4 out5 out6] = gaga_matlab_gen(alg,in3,in4,in5);
    elseif (nargin == 6)
      [out1 out2 out3 out4 out5 out6] = gaga_matlab_gen(alg,in3,in4,in5,in6);
    else
      error('Either five or six input arguments required.');
    end
elseif strcmp(ens,'smv')
   % [norms times iterations support convRate vec_out] = gaga_smv(algstring,k,m,n,p,options)
      if (nargin == 6)
        [out1 out2 out3 out4 out5 out6] = gaga_matlab_smv(alg,in3,in4,in5,in6);
      elseif (nargin == 7)
        [out1 out2 out3 out4 out5 out6] = gaga_matlab_smv(alg,in3,in4,in5,in6,in7);
      else
        error('Either six or seven input arguments required.');
      end
elseif strcmp(ens,'dct')
  % [norms times iterations support convRate vec_out] = gaga_dct(algstring,k,m,n,options)
    if (nargin == 5)
      [out1 out2 out3 out4 out5 out6] = gaga_matlab_dct(alg,in3,in4,in5);
    elseif (nargin == 6)
      [out1 out2 out3 out4 out5 out6] = gaga_matlab_dct(alg,in3,in4,in5,in6);
    else
      error('Either five or six input arguments required.');
    end
else
  error('ens must be either dct, gen, or smv.')
end



      




