TARGET=FYArray.exe
FFLAGS = 
CXX=icpc -qopenmp -xCORE-AVX512 -qopt-zmm-usage=high -fp-model fast=2 $(FFLAGS)
#-qopt-report=5#-xavx2 -Ofast -fno-alias

INCLUDE=../include

OBJ=FYStorage.o main.o

$(TARGET):$(OBJ)
	$(CXX) $^ -o $@ 
	cp $(TARGET) ../

%.o:%.cpp
	$(CXX) -I$(INCLUDE) -c $< -o $@

.PHONY:clean
clean:
	rm *.o -f
