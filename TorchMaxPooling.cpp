#include "TorchLoader.h"
#include "TorchMaxPooling.h"
#include "TorchTable.h"

namespace NN
{
  TorchMaxPooling::TorchMaxPooling()
  {
  }

  std::shared_ptr<TorchObject> TorchMaxPooling::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int tableId = readNextInt(file);
    (void)TorchLoader::getInstance()->create(tableId, file, loaded);
    return std::shared_ptr<TorchObject>(this);
  }
}
