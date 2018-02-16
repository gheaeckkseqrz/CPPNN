#ifndef __TORCHLOADER_H__
#define __TORCHLOADER_H__

#include "TorchObject.h"

namespace NN
{
  class TorchLoader
  {
  public:
    static TorchLoader *getInstance();

    TorchObject *loadFile(std::string const &torchFilePath);
    TorchObject *create(int objectType, std::ifstream &file, std::map<int, TorchObject*> &loaded);

  private:
    TorchLoader();
    static TorchLoader *_instance;
  };
}
#endif /* __TORCH_LOADER_H__ */
