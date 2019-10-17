GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=0
AVX=0
OPENMP=0
LIBSO=1

# set GPU=1 and CUDNN=1 to speedup on GPU
# set CUDNN_HALF=1 to further speedup 3 x times (Mixed-precision on Tensor Cores) GPU: Volta, Xavier, Turing and higher
# set AVX=1 and OPENMP=1 to speedup on CPU (if error occurs then set AVX=0)

DEBUG=0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
	  -gencode arch=compute_61,code=[sm_61,compute_61]

OS := $(shell uname)

# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070	Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000	Tesla T4
# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# Jetson XAVIER
# ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 � DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]


OBJDIR=./obj/
BASE_LIB_OBJ_DIR=./obj/base/
BASE_SRC_DIR=./darknet/src/
OVERRIDE_LIB_OBJ_DIR=./obj/override/
OVERRIDE_SRC_DIR=./src/
LIBNAMESO=libdarknet.so

CC=gcc
NVCC=nvcc 
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC

ifeq ($(DEBUG), 1) 
OPTS= -O0 -g
else
ifeq ($(AVX), 1) 
CFLAGS+= -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a
endif
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(OPENMP), 1)
CFLAGS+= -fopenmp
LDFLAGS+= -lgomp
endif

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
ifeq ($(OS),Darwin) #MAC
LDFLAGS+= -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand
else
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif
endif

ifeq ($(CUDNN), 1)
COMMON+= -DCUDNN
ifeq ($(OS),Darwin) #MAC
CFLAGS+= -DCUDNN -I/usr/local/cuda/include
LDFLAGS+= -L/usr/local/cuda/lib -lcudnn
else
CFLAGS+= -DCUDNN -I/usr/local/cudnn/include
LDFLAGS+= -L/usr/local/cudnn/lib64 -lcudnn
endif
endif

ifeq ($(CUDNN_HALF), 1)
COMMON+= -DCUDNN_HALF
CFLAGS+= -DCUDNN_HALF
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]
endif

BASE_LIB_OBJ=activations.o \
activation_layer.o \
art.o \
avgpool_layer.o \
batchnorm_layer.o \
blas.o \
box.o \
captcha.o \
cifar.o \
classifier.o \
col2im.o \
compare.o \
connected_layer.o \
convolutional_layer.o \
cost_layer.o \
cpu_gemm.o \
crnn_layer.o \
crop_layer.o \
data.o \
deconvolutional_layer.o \
detection_layer.o \
dice.o \
dropout_layer.o \
gemm.o \
getopt.o \
go.o \
gru_layer.o \
im2col.o \
image.o \
layer.o \
list.o \
local_layer.o \
matrix.o \
maxpool_layer.o \
network.o \
nightmare.o \
normalization_layer.o \
option_list.o \
region_layer.o \
reorg_layer.o \
reorg_old_layer.o \
rnn.o \
rnn_layer.o \
rnn_vid.o \
route_layer.o \
shortcut_layer.o \
softmax_layer.o \
super.o \
swag.o \
tag.o \
tree.o \
upsample_layer.o \
voxel.o \
yolo_layer.o \
lstm_layer.o \
sam_layer.o \
conv_lstm_layer.o \
scale_channels_layer.o

OVERRIDE_LIB_OBJ=darknet.o \
cuda.o \
error.o \
utils.o

ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

BASE_LIB_OBJS=$(addprefix $(BASE_LIB_OBJ_DIR), $(BASE_LIB_OBJ))
BASE_LIB_DEPS=$(wildcard $(BASE_SRC_DIR)/*.h) Makefile
BASE_LIB_CFLAGS=$(CFLAGS) -I darknet/include -I darknet/3rdparty/stb/include
OVERRIDE_LIB_OBJS=$(addprefix $(OVERRIDE_LIB_OBJ_DIR), $(OVERRIDE_LIB_OBJ))
OVERRIDE_LIB_DEPS=$(wildcard ./include/*.h)
OVERRIDE_LIB_CFLAGS=$(CFLAGS) -I darknet/include

all: obj backup results $(LIBNAMESO)

$(LIBNAMESO): $(BASE_LIB_OBJS) $(OVERRIDE_LIB_OBJS)
	$(CC) -shared -DYOLODLL_EXPORTS $(COMMON) $(CFLAGS) $(BASE_LIB_OBJS) $(OVERRIDE_LIB_OBJS) -o $@ $(LDFLAGS)

$(BASE_LIB_OBJ_DIR)%.o: $(BASE_SRC_DIR)%.c $(BASE_LIB_DEPS)
	$(CC) $(COMMON) ${BASE_LIB_CFLAGS} -c $< -o $@

$(BASE_LIB_OBJ_DIR)%.o: $(BASE_SRC_DIR)%.cu $(BASE_LIB_DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

$(OVERRIDE_LIB_OBJ_DIR)%.o: $(OVERRIDE_SRC_DIR)%.c $(OVERRIDE_LIB_DEPS)
	$(CC) $(COMMON) $(OVERRIDE_LIB_CFLAGS) -c $< -o $@

obj:
	echo $(BASE_LIB_OBJ_DIR)
	echo $(OVERRIDE_LIB_OBJ_DIR)
	mkdir -p obj
	mkdir -p $(BASE_LIB_OBJ_DIR)
	mkdir -p $(OVERRIDE_LIB_OBJ_DIR)
backup:
	mkdir -p backup
results:
	mkdir -p results
.PHONY: clean

clean:
	rm -rf $(OVERRIDE_LIB_OBJS) $(BASE_LIB_OBJS) $(LIBNAMESO)
