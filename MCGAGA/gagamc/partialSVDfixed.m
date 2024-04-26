function [U,S,V,I]=partialSVDfixed(A,r,tol,mxIter)

if nargin<4
    mxIter=50;
    if nargin<3
        tol = 10^(-6);  % find the singular vectors to single precision
        if nargin < 2
            r = 1;      % if rank is not given, find only the top singular vector
        end
    end
end

[m,n]=size(A);
err_Scale=1/sqrt(m);
u=zeros(m,1);
v=rand(n,1)/sqrt(n);
s=0;
iter=0;

U=zeros(m,r);
S=zeros(r,1);
V=zeros(n,r);
I=zeros(r,1);

err=tol+1;

for i=1:r
  while ( (err>tol) && (iter<mxIter) )
    u_prev=u;
    u=A*v; u=u/norm(u);
    v=A'*u;
    s=norm(v);
    v=v/s;
    err=norm(u-u_prev)*err_Scale;
    iter=iter+1;
  end

  U(:,i)=u;
  S(i)=s;
  V(:,i)=v;
  I(i)=iter;

  A=A-s*u*v';
  
  u=zeros(m,1);
  v(1+mod(i,2):2:end)=randn(floor(n/2)+mod(n,2),1)/sqrt(n);
  err=tol+1;
  iter=0;
end
    


