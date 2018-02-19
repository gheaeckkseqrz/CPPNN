#ifndef __TORCHRELU_H__
#define __TORCHRELU_H__

#include "Relu.h"
#include "TorchObject.h"

namespace NN
{
  class TorchRelu : public Relu, public TorchObject
  {
  public:
    TorchRelu();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
  };
}

#endif /* __TORCHRELU_H__ */
