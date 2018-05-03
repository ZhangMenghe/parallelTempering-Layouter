CC = g++
NVCC = nvcc
CUDAFLAGS = -Wno-deprecated-gpu-targets -arch=compute_30
CXXFLAGS = -g -Wall
PATH_INCLUDE = "C:/Projects/parallelTempering-Layouter/include"
PATH_LIB = "C:/Projects/parallelTempering-Layouter/lib"
REFLIBS =  -lopencv_core340 -lopencv_highgui340 -lopencv_imgproc340 -lopencv_imgcodecs340 -lopencv_videoio340
LINK_TARGET = mcmc.exe
# OBJS = hello.o
# DEPS = hello.h
SRCS = mcmc.cu room.cu

REBUILD_ABLES = $(OBJS) $(LINK_TARGET)
all: $(LINK_TARGET)
	echo all done
# $(LINK_TARGET):$(OBJS)
# 	$(CC) $(OBJS) -o $@
# specify INCLUDE and LIB directory
$(LINK_TARGET): mcmc.cu room.cu
	$(NVCC) $(CUDAFLAGS) $(SRCS) -o $@
# $(LINK_TARGET): mcmc.cu room.cu
# 	$(NVCC) $(CUDAFLAGS) -I$(PATH_INCLUDE) -L$(PATH_LIB) $(REFLIBS) $(SRCS) -o $@
clean:
	 del -f *.h.gch *.o *.exe *.exp *.lib
	 echo clean done
