#ifndef __PCA_H__
#define __PCA_H__

#include <Tensor.h>

namespace NN
{
  class GPUOp
  {
  public:
    GPUOp(std::shared_ptr<Tensor> t);

    int rows();
    int cols();
    void perform_op(float *x_in, float *y_out);

  private:
    std::shared_ptr<Tensor> _in;
    std::shared_ptr<Tensor> _out;
    std::shared_ptr<Tensor> _t;
  };

  class PCA
  {
    public:
      PCA();

      std::shared_ptr<Tensor> reduceDataToNbOfDims(std::shared_ptr<Tensor> data, int dims);
  };
}

#endif /* __PCA_H__ */
