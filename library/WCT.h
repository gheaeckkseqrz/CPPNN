#ifndef __WCT_H__
#define __WCT_H__

#include "Tensor.h"

namespace NN
{
  class WCT
  {
  public:
    WCT();

    std::shared_ptr<Tensor> enforceCovariance(std::shared_ptr<Tensor> data, std::shared_ptr<Tensor> targetCovariance, std::shared_ptr<Tensor> targetMean);
    std::shared_ptr<Tensor> whitenData(std::shared_ptr<Tensor> data);
  };
}

#endif /* __WCT_H__ */
