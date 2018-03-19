#include "TorchLoader.h"
#include "TorchNumber.h"
#include "TorchReflectionPadding.h"
#include "TorchTable.h"

namespace NN
{
  TorchReflectionPadding::TorchReflectionPadding()
  {
  }

  std::shared_ptr<TorchObject> TorchReflectionPadding::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int tableId = readNextInt(file);
    std::shared_ptr<TorchTable> table = std::dynamic_pointer_cast<TorchTable>(TorchLoader::getInstance()->create(tableId, file, loaded));

    _padT = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("pad_t"))->value();
    _padB = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("pad_b"))->value();
    _padR = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("pad_r"))->value();
    _padL = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("pad_l"))->value();
    return std::shared_ptr<TorchObject>(this);
  }
}
