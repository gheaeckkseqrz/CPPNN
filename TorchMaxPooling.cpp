#include "TorchLoader.h"
#include "TorchMaxPooling.h"
#include "TorchTable.h"

namespace NN
{
  TorchMaxPooling::TorchMaxPooling()
  {
  }

  TorchObject *TorchMaxPooling::loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded)
  {
    int tableId = readNextInt(file);
    TorchTable *t = (TorchTable*)TorchLoader::getInstance()->create(tableId, file, loaded);
    return this;
  }
}
