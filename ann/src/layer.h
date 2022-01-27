#ifndef _LAYER_H_
#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.h"
#include "loss.h"
#include "helper.h"

namespace betterdl
{
class Layer
{
public:
  Layer();
  virtual ~Layer();

  virtual Blob<float> *forward(Blob<float> *input) = 0;

  // 当前层的backwad输入的是前一个层的grad
  virutal Blob<float> *backward(Blob<float> *grad_input) = 0;

  std::string get_name() {
    return name_;
  }

  virtual float get_loss(Blob<float> *target); // todo 入参不理解
  virtual int get_accuracy(Blob<float> *target);// todo 函数作用是什么

  void set_cuda_context(CudaContext *context) { cuda_ = context;  } // todo context是啥

  void set_load_pretrain() {
    load_pretrain_ = true;
  }

  void set_gradient_stop() {
    gradient_stop_ = true;
  }

  void freeze() {// todo when to use
    freeze_ = true;
  }

  void unfreeze() {
    freeze_ = false;
  }

protected:
  virtual void fwd_initialize(Blob<float> *input) = 0;
  virtual void bwd_initialize(Blob<float> *grand_output) = 0;

  void init_weight_bias(unsigned int seed = 0);
  void update_weights_biases(float learning_rate);

  int load_parameter();
  int save_parameter();


  // name of layer
  std::string name_;

  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;

  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t bias_desc_;

  // output memory
  Blob<float> *input_ = nullptr;// x
  Blob<float> *output_ = nullptr;//y
  Blob<float> *grad_input_ = nullptr;// dx
  Blob<float> *grad_output_= nullptr;// dy

  //master weights & bias
  bool freeze_ = false; // control parameter updates
  Blob<float> *weights_ = nullptr; // w
  Blob<float> *biases_ = nullptr;// b
  Blob<float> *grad_weights_ = nullptr; // dw
  Blob<float> *grad_biases_ = nullptr; // dg

  int batch_size_ = 0;

  CudaContext *cuda_ = nullptr;

  bool load_pretrain_ = false;

  bool gradient_stop_ = false;

  friend class Network;
};

class Dense: public Layer
{
public:
  Dense(std::string name, int out_size);
  virtual ~Dense();

  virtual Blob<float> *forward(Blob<float> *intput);
  virtual Blob<float> *backward(Blob<float> *grad_input);

private:
  void fwd_initialize(Blob<float> *input);
  void bwd_initialize(Blob<float> *grad_output);

private:
  int input_size_ = 0;
  int output_size_ = 0;
  float *d_one_vec = nullptr;
};

class Activation: public Layer
{
public:
  Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);
  virtual ~Activation();

  virtual Blob<float> *forward(Blob<float> *input);
  virtual Blob<float> *backward(Blob<float> *grad_input);

private:
  void fwd_initialize(Blob<float> *input);
  void bwd_initialize(Blob<falot> *grad_output);

  cudnnActivationDescriptor_t act_desc_;
  cudnnActivationMode_t act_mode_;
  float act_coef_;
};

class Softmax: public Layer
{
    public:
    Softmax(std::string name);
    virtual ~Softmax();

    virtual Blob<float> *forward(Blob<float> *input);
    virtual Blob<float> *backward(Blob<float> *grad_input);

    float get_loss(Blob<float> *target);
    int   get_accuracy(Blob<float> *target);

    protected:
    void fwd_initialize(Blob<float> *input);
    void bwd_initialize(Blob<float> *grad_output);

    CrossEntropyLoss loss_;
};
}
