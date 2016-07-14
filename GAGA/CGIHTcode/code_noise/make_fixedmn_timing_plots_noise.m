% make timing plot for fixed  m and n with respect to k

function make_fixedmn_timing_plots_noise(alg_list, matens, n, noise_level, nonzeros)

fontsz=16;


noise_string = sprintf('_noise%0.3f',noise_level);
if (noise_level == 0)
  noise_string = '';
end

fname_noise_string = strrep(noise_string,'.','-');

dd_list=linspace(0.1,.99,20);
[~,d_ind]=min(abs(dd_list-0.3));

delta_list = [0.1 dd_list(d_ind)];

delta_list=[0.1 0.3]; % reduction for representative data, delete for full data

for dd=1:length(delta_list)
delta=delta_list(dd);
m=ceil(n*delta); 



[c_list s_list label_list]=set_color_symbol(alg_list);

% useage of make_fixedmn_timing_plot
% [k_list, ave_time_list] = make_fixedmn_timing_plot_joint(alg, ens, m, n, l, succ_prob, nonzeros)

plotvalues=cell(length(alg_list),2);

if nargin == 3
  nonzeros = [];
end
succ_prob = 0.8;
figure();
hold off


for jj=1:length(alg_list)
  [plotvalues{jj,1}, plotvalues{jj,2}] =  make_fixedmn_rho_time_list_noise(alg_list{jj}, matens, m, n, succ_prob, noise_level, nonzeros);
  if strcmp(alg_list{jj},'CGIHTprojected')
    semilogy(plotvalues{jj,1}/m, plotvalues{jj,2}, 'color', [1.0 0.6 0]);
  else
    semilogy(plotvalues{jj,1}/m, plotvalues{jj,2}, c_list{jj});
  end
  hold on
end

leg=legend(alg_list{:},'Location','NorthWest');
set(leg,'FontSize',fontsz)



axis tight
set(gca,'Fontsize',fontsz);

if strcmp(matens,'gen')
  enstxt = '{\fontname{zapfchancery}N}';
elseif strcmp(matens,'smv')
  enstxt = ['{\fontname{zapfchancery}S}_{' num2str(nonzeros) '}'];
else enstxt = 'DCT';
end

if noise_level == 0
  ProbClass=['(' enstxt ',B)'];
else
  ProbClass=['(' enstxt ',B_\epsilon),' sprintf(' \\epsilon = %0.1f',noise_level)];
end

title(['Average recovery time (ms) for ' ProbClass ': m = ', num2str(m), ', n = ', num2str(n)],'Fontsize',fontsz)


xlabel('\rho=k/m','Fontsize',fontsz)
ylabel('Time(ms)','Fontsize',fontsz)

fname_out=['noiseplots/ave_time_' matens '_m_' num2str(m) '_n_' num2str(n)]; 
if strcmp(matens, 'smv') == 1
  fname_out=[fname_out '_nonzeros_' num2str(nonzeros)];
end
fname_out=[fname_out fname_noise_string '.pdf'];
print('-dpdf',fname_out)


end % ends the delta_list loop


%%%%%%%%%%% Other way to create delta values %%%%%%%%%%%%5

%{
if strcmp(matens, 'gen') == 1 && n == 2^12
  delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04];
  delta_list=sort(delta_list);
end

if (strcmp(matens, 'dct') == 1 && n == 2 ^20)
    delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
    delta_list=sort(delta_list);
end

if (strcmp(matens, 'smv') == 1 && n == 2^18)
    delta_list=[linspace(0.1,0.99,20) 0.08 0.06 0.04 0.02 0.01 0.008 0.006 0.004 0.002 0.001];
    delta_list=sort(delta_list);
end


m_list=ceil(n*delta_list); % must be the same with generate_data...
m_list=sort(m_list); 

%delta_selected = input('Specify a delta to do rho ~ time plot for fixed m, n: ');
delta_selected = 0.3;
[~,ind]=sort(abs(delta_list-delta_selected),'ascend');
delta_in_use = delta_list(ind(1));
m = ceil(n*delta_in_use);
if any(m==m_list) == 0
  error('No such m = %d in the m_list', m)
end
%}

%%%%%%%%%%%%%%%% Other way to create legend %%%%%%%%%%%%%
%{
switch length(alg_list)
  case 1
    legend(alg_list{1},'location', 'northwest','Fontsize',fontsz)
  case 2
    legend(alg_list{1},alg_list{2},'location', 'northwest','Fontsize',fontsz)
  case 3
    legend(alg_list{1},alg_list{2},alg_list{3},'location', 'northwest','Fontsize',fontsz)
  case 4
    legend(alg_list{1},alg_list{2},alg_list{3},alg_list{4},'location', 'northwest','Fontsize',fontsz)
  case 5
    legend(alg_list{1},alg_list{2},alg_list{3},alg_list{4},alg_list{5},'location', 'northwest','Fontsize',fontsz)
  case 6
    legend(alg_list{1},alg_list{2},alg_list{3},alg_list{4},alg_list{5},alg_list{6},'location', 'northwest','Fontsize',fontsz)
  case 7
    legend(alg_list{1},alg_list{2},alg_list{3},alg_list{4},alg_list{5},alg_list{6},alg_list{7},'location', 'northwest','Fontsize',fontsz)
  case 8
    legend(alg_list{1},alg_list{2},alg_list{3},alg_list{4},alg_list{5},alg_list{6},alg_list{7},alg_list{8},'location', 'northwest','Fontsize',fontsz)
  case 9
    legend(alg_list{1},alg_list{2},alg_list{3},alg_list{4},alg_list{5},alg_list{6},alg_list{7},alg_list{8},alg_list{9},'location', 'northwest','Fontsize',fontsz)
  case 10
    legend(alg_list{1},alg_list{2},alg_list{3},alg_list{4},alg_list{5},alg_list{6},alg_list{7},alg_list{8},alg_list{9},alg_list{10},'location', 'northwest','Fontsize',fontsz)
  otherwise
    warning('The legend needs to be built directly');
end
%}


