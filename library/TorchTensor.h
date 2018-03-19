#ifndef __TORCHTENSOR_H__
#define __TORCHTENSOR_H__

#include "Tensor.h"
#include "TorchObject.h"

namespace NN
{
  class TorchTensor : public Tensor, public TorchObject
  {
  public:
    TorchTensor();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
  };
}

#endif /* __TORCHTENSOR_H__ */
