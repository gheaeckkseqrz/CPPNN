#ifndef __TORCHSEQUENTIAL_H__
#define __TORCHSEQUENTIAL_H__

#include "TorchObject.h"
#include "Sequential.h"

namespace NN
{
  class TorchSequential : public Sequential, public TorchObject
  {
  public:
    TorchSequential();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
  };
}

#endif /* __TORCHSEQUENTIAL_H__ */
