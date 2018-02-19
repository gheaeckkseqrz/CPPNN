#include "TorchRelu.h"
#include "TorchLoader.h"
#include "TorchTable.h"

namespace NN
{
  TorchRelu::TorchRelu()
  {
  }

  std::shared_ptr<TorchObject> TorchRelu::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int tableId = readNextInt(file);
    (void)TorchLoader::getInstance()->create(tableId, file, loaded);
    return std::shared_ptr<TorchObject>(this);
  }
}
