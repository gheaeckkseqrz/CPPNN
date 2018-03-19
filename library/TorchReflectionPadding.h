#ifndef __TORCHREFLECTIONPADDING_H__
#define __TORCHREFLECTIONPADDING_H__

#include "ReflectionPadding.h"
#include "TorchObject.h"

namespace NN
{
  class TorchReflectionPadding : public ReflectionPadding, public TorchObject
{
 public:
  TorchReflectionPadding();

  virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
};
}

#endif /* __TORCHREFLECTIONPADDING_H__ */
