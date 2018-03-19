#ifndef __TORCHSTORAGE_H__
#define __TORCHSTORAGE_H__

#include "TorchObject.h"
#include "Storage.h"

namespace NN
{
  class TorchStorage : public TorchObject, public Storage<float>
  {
  public:
    TorchStorage();
    virtual ~TorchStorage() {}

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
  };
}

#endif /* __TORCHSTORAGE_H__ */
