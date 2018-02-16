#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include "Module.h"
#include "Tensor.h"

namespace NN
{
  class Convolution : public Module
  {
  public:
    Convolution();
    ~Convolution();

    void setFilter(std::shared_ptr<Tensor> const &filter, std::shared_ptr<Tensor> const &bias = nullptr);
    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input);

  protected:
    std::shared_ptr<Tensor> _filter;
    std::shared_ptr<Tensor> _bias;
    int _padW;
    int _padH;
    int _dW;
    int _dH;
  };
}

#endif /* __CONVOLUTION_H__ */
