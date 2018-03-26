#ifndef __TENSORFLOW_CONVOLUTION_H__
#define __TENSORFLOW_CONVOLUTION_H__

#include "Convolution.h"
#include "Node.h"

namespace NN
{
  class TensorflowConvolution : public Node
  {
  public:
    TensorflowConvolution(std::list<std::string> const &inputs);

    virtual std::shared_ptr<Tensor> evaluate(ComputeGraph const &graph);

  private:
    std::shared_ptr<Tensor> flipFilter(std::shared_ptr<Tensor> const &filter);
  };
};

#endif
