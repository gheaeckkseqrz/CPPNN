#include "TorchRelu.h"
#include "TorchLoader.h"
#include "TorchTable.h"

namespace NN
{
  TorchRelu::TorchRelu()
  {
  }

  TorchObject *TorchRelu::loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded)
  {
    int tableId = readNextInt(file);
    TorchTable *t = (TorchTable*)TorchLoader::getInstance()->create(tableId, file, loaded);
    return this;
  }
}
