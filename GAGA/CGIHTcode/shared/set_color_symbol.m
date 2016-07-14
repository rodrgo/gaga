function [color_list symbol_list label_list] = set_color_symbol(alg_list)


color_list=cell(length(alg_list),1);
symbol_list=cell(length(alg_list),1);
label_list=cell(length(alg_list),1);

for jj=1:length(alg_list)
  switch alg_list{jj}
    case 'NIHT'
      color_list{jj}='k';
      symbol_list{jj}='o';
      label_list{jj}=['NIHT: circle'];
    case 'CGIHTrestarted'
      color_list{jj}='r';
      symbol_list{jj}='+';
      label_list{jj} = ['CGIHTrestarted: plus'];
    case 'CGIHT'
      color_list{jj}='m';  
      symbol_list{jj}='d';
      label_list{jj} = ['CGIHT: diamond'];
    case 'CGIHTprojected'
      color_list{jj}='y';  %cell(2,1); color_list{jj}{1}='color'; color_list{jj}{2}=[1.0,0.6,0];
      symbol_list{jj}='h';
      label_list{jj} = ['CGIHTprojected: hexagon'];
    case 'CSMPSP'
      color_list{jj}='b';
      symbol_list{jj}='s';
      label_list{jj} = ['CSMPSP: square'];
    case 'HTP'
      color_list{jj}='c';
      symbol_list{jj}='*';
      label_list{jj} = ['HTP: asterisk'];
    case 'FIHT'
      color_list{jj}='g';
      symbol_list{jj}='x';
      label_list{jj} = ['FIHT: times'];
    otherwise
      color_{list} = 'y';
      symbol_list{jj}='*';
      label_list{jj} = [alg_list{jj} ': asterisk'];
  end
end

     
