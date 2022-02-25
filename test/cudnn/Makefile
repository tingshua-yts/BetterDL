CXX := nvcc
TARGET := test_conv
CUDNN_PATH := cudnn
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib -L/usr/lib64
CXXFLAGS := -arch=sm_35 -std=c++11 -O2


conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcudnn `pkg-config --libs opencv`

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
