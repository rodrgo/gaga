function v = AT_dct(u, m, n, dct_rows)
 
  utemp = zeros(n,1);
  utemp(dct_rows) = u;
  v = idct2(utemp);
  return;
