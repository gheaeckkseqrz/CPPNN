#ifndef __SINGULARVALUEDECOMPOSITION_H__
#define __SINGULARVALUEDECOMPOSITION_H__

#include "Tensor.h"

namespace NN
{
  class SingularValueDecomposition
  {
  public:
    SingularValueDecomposition(std::shared_ptr<Tensor> matrix);

    std::shared_ptr<Tensor> getU() const;
    std::shared_ptr<Tensor> getValue() const;

  private:
    std::shared_ptr<Tensor> _u;
    std::shared_ptr<Tensor> _v;
  };
}

#endif /* __SINGULARVALUEDECOMPOSITION_H__ */
