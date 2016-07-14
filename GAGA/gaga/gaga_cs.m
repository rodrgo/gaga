function [out1 out2 out3 out4 out5 out6 resRecord timeRecord] = gaga_cs(alg, ens, in3, ...
                                                  in4, in5, in6, ...
                                                  in7, in8)

% Need to add help info...

if strcmp(ens,'gen')
  if (prod(size(in4)) > 1) % [outputVector iterations convRate] = gaga_gen(algstring,y,A,k,options)  
    if (nargin == 5) 
      [out1 out2 out3] = gaga_gen(alg,in3,in4,in5);
    elseif (nargin == 6)
      [out1 out2 out3] = gaga_gen(alg,in3,in4,in5,in6);
    else
      error('Either five or six input arguments required.');
    end
  else % [norms times iterations support convRate xout] = gaga_gen(algstring,k,m,n,options)
    if (nargin == 5)
      [out1 out2 out3 out4 out5 out6] = gaga_gen(alg,in3,in4,in5);
    elseif (nargin == 6)
      [out1 out2 out3 out4 out5 out6] = gaga_gen(alg,in3,in4,in5,in6);
    else
      error('Either five or six input arguments required.');
    end
  end  
elseif strcmp(ens,'smv')
  if (prod(size(in5)) > 1) % [outputVector iterations convRate resRecord timeRecord] = gaga_smv(algstring,y,smv_rows,smv_cols,smv_vals,k,options)
    % out4 and out5 are resRecord and timeRecord
    if (nargin == 7)
      [out1 out2 out3 out4 out5] = gaga_smv(alg,in3,in4,in5,in6,in7);
    elseif (nargin == 8)
      [out1 out2 out3 out4 out5] = gaga_smv(alg,in3,in4,in5,in6,in7,in8);
    else
      error('Either seven or eight input arguments required.');
    end
    else % [norms times iterations support convRate vec_out resRecord timeRecord] = gaga_smv(algstring,k,m,n,p,options)
      if (nargin == 6)
        [out1 out2 out3 out4 out5 out6 resRecord timeRecord] = gaga_smv(alg,in3,in4,in5,in6);
      elseif (nargin == 7)
        [out1 out2 out3 out4 out5 out6 resRecord timeRecord] = gaga_smv(alg,in3,in4,in5,in6,in7);
      else
        error('Either six or seven input arguments required.');
      end
  end
elseif strcmp(ens,'dct')
  if (prod(size(in4)) > 1) % [outputVector iterations convRate] = gaga_dct(algstring,y,dct_rows,k,m,n,options)
    if (nargin == 7)
      [out1 out2 out3] = gaga_dct(alg,in3,in4,in5,in6,in7);
    elseif (nargin == 8)
      [out1 out2 out3] = gaga_dct(alg,in3,in4,in5,in6,in7,in8);
    else
      error('Either seven or eight input arguments required.');
    end
  else % [norms times iterations support convRate vec_out] = gaga_dct(algstring,k,m,n,options)
    if (nargin == 5)
      [out1 out2 out3 out4 out5 out6] = gaga_dct(alg,in3,in4,in5);
    elseif (nargin == 6)
      [out1 out2 out3 out4 out5 out6] = gaga_dct(alg,in3,in4,in5,in6);
    else
      error('Either five or six input arguments required.');
    end
  end
else
  error('ens must be either dct, gen, or smv.')
end



      




