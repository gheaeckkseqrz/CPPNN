#ifndef __TENSORFLOW_TENSOR_H__
#define __TENSORFLOW_TENSOR_H__

#include "Tensor.h"

namespace NN
{
  class TensorflowTensor : public Tensor
  {
  public:
    TensorflowTensor();
    TensorflowTensor(std::string const &s);

    void init(std::string const &s);
    float parseFloat(std::string const &s, int &index);
  };
}

#endif
