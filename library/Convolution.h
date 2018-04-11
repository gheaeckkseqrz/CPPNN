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
    void setPadding(int padW, int padH);
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> const &input);
    virtual std::string print() const;

  protected:
    void processBlock(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> im2col, std::pair<int, int> slice, int kernelH, int kernelW);

  protected:
    std::shared_ptr<Tensor> _filter;
    std::shared_ptr<Tensor> _bias;
    int _padW;
    int _padH;
    int _dW;
    int _dH;
    int _dilationW;
    int _dilationH;
  };
}

#endif /* __CONVOLUTION_H__ */
