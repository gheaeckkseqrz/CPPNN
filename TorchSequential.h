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

    virtual TorchObject *loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded);
  };
}

#endif /* __TORCHSEQUENTIAL_H__ */
