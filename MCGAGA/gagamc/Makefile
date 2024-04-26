include ../config  

#VERSION=$(shell date +"%Y%m%d")
VERSION="_0_0_0"

TARGET_ONE = gagamc_entry_v2.mexa64

TARGET_TWO = gagamc_entry_suic.mexa64

TARGET_SVD1 = gpuPartialSVD_v2.mexa64

TARGET_SVD2 = gpuPartialSVD_SPI.mexa64

TARGET_PACKAGE = MCGAGA$(VERSION).tar


.PHONY: all clean_all 


all: $(TARGET_ONE) $(TARGET_TWO) $(TARGET_SVD1) $(TARGET_SVD2)

clean_all:
	rm -f $(TARGET_ONE) $(TARGET_TWO) $(TARGET_SVD1) $(TARGET_SVD2)

one: $(TARGET_ONE)

clean_one: 
	rm -f $(TARGET_ONE)

two: $(TARGET_TWO)

clean_two: 
	rm -f $(TARGET_TWO)

package: $(TARGET_PACKAGE)

clean_package:
	rm -f ../../$(TARGET_PACKAGE)



%.mexa64: %.cu
	$(MAKE) -f $(CUDAMATLAB)/Makefile $@

%.tar:
	cd $(GAGA)/.. && tar -cvf MCGAGA$(VERSION).tar -X $(GAGA)/exclude MCGAGA

