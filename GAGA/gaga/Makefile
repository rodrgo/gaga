include ../config  

#VERSION=$(shell date +"%Y%m%d")
VERSION="_1_1_9"

TARGET_ONE=gaga_gen.mexa64


TARGET_DCT=gaga_dct.mexa64 

TARGET_SMV=gaga_smv.mexa64 

TARGET_GEN=gaga_gen.mexa64 

TARGET_PACKAGE=GAGA$(VERSION).tar


.PHONY: all clean_all clean_smv smv clean_dct dct 

all: $(TARGET_DCT) $(TARGET_SMV) $(TARGET_GEN) 

clean_all:
	rm -f $(TARGET_DCT) $(TARGET_SMV) $(TARGET_GEN) 

one: $(TARGET_ONE)

clean_one: 
	rm -f $(TARGET_ONE)

dct: $(TARGET_DCT)

clean_dct:
	rm -f $(TARGET_DCT)

smv: $(TARGET_SMV)

clean_smv:
	rm -f $(TARGET_SMV)

gen: $(TARGET_GEN)

clean_gen:
	rm -f $(TARGET_GEN)

package: $(TARGET_PACKAGE)

clean_package:
	rm -f ../../$(TARGET_PACKAGE)



%.mexa64: %.cu
	$(MAKE) -f $(CUDAMATLAB)/Makefile $@

%.tar:
	cd $(GAGA)/.. && tar -cvf GAGA$(VERSION).tar -X $(GAGA)/exclude GAGA && tar -rvf GAGA$(VERSION).tar GAGA/UserGuide_1_1_0.pdf GAGA/gaga_for_CS_timings/GAGA_table_and_plots.tex
        

