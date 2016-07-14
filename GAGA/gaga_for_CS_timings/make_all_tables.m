tic
display('Making tables started');

make_combined_table('NIHT','dct',0,1,0)
make_combined_table('NIHT','gen',0,1,0)
make_combined_table('NIHT','smv',0,1,0)

display(sprintf('Tables for steepest descent times completed in %f seconds', toc));

make_combined_table('HTP','dct',0,3,0)
make_combined_table('HTP','gen',0,3,0)
make_combined_table('HTP','smv',0,3,0)

display(sprintf('Tables for cg times completed in %f seconds', toc));

make_combined_table('NIHT','dct',0,5,0)
make_combined_table('NIHT','gen',0,5,0)
make_combined_table('NIHT','smv',0,5,0)

display(sprintf('Tables for problem generation times completed in %f seconds', toc));


make_acceleration_table
display(sprintf('Acceleration tables completed in %f seconds', toc));

display('Making tables for paper completed');


return

display('Making tables for appendix');

make_table('NIHT','gpu','dct',0,1,0)
make_table('NIHT','matlab','dct',0,1,0)
make_table('NIHT','gpu','gen',0,1,0)
make_table('NIHT','matlab','gen',0,1,0)
make_table('NIHT','gpu','smv',0,1,0,4)
make_table('NIHT','matlab','smv',0,1,0,4)
make_table('NIHT','gpu','smv',0,1,0,7)
make_table('NIHT','matlab','smv',0,1,0,7)
make_table('NIHT','gpu','smv',0,1,0,13)
make_table('NIHT','matlab','smv',0,1,0,13)

make_table('HTP','gpu','dct',0,3,0)
make_table('HTP','matlab','dct',0,3,0)
make_table('HTP','gpu','gen',0,3,0)
make_table('HTP','matlab','gen',0,3,0)
make_table('HTP','gpu','smv',0,3,0,4)
make_table('HTP','matlab','smv',0,3,0,4)
make_table('HTP','gpu','smv',0,3,0,7)
make_table('HTP','matlab','smv',0,3,0,7)
make_table('HTP','gpu','smv',0,3,0,13)
make_table('HTP','matlab','smv',0,3,0,13)

make_table('NIHT','gpu','dct',0,5,0)
make_table('NIHT','matlab','dct',0,5,0)
make_table('NIHT','gpu','gen',0,5,0)
make_table('NIHT','matlab','gen',0,5,0)
make_table('NIHT','gpu','smv',0,5,0,4)
make_table('NIHT','matlab','smv',0,5,0,4)
make_table('NIHT','gpu','smv',0,5,0,7)
make_table('NIHT','matlab','smv',0,5,0,7)
make_table('NIHT','gpu','smv',0,5,0,13)
make_table('NIHT','matlab','smv',0,5,0,13)

display('Making tables for appendix completed');

  
