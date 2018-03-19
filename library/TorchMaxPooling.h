#ifndef __TORCHMAXPOOLING_H__
#define __TORCHMAXPOOLING_H__

#include "MaxPooling.h"
#include "TorchObject.h"

namespace NN
{
  class TorchMaxPooling : public MaxPooling, public TorchObject
  {
  public:
    TorchMaxPooling();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
  };
}

#endif /* __TORCHMAXPOOLING_H__ */
