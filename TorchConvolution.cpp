#include "TorchConvolution.h"
#include "TorchLoader.h"
#include "TorchNumber.h"
#include "TorchTable.h"
#include "TorchTensor.h"

namespace NN
{
  TorchConvolution::TorchConvolution()
  {
  }

  std::shared_ptr<TorchObject> TorchConvolution::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int tableId = readNextInt(file);
    std::shared_ptr<TorchTable> table = std::dynamic_pointer_cast<TorchTable>(TorchLoader::getInstance()->create(tableId, file, loaded));

    std::shared_ptr<Tensor> filterPtr = std::dynamic_pointer_cast<Tensor>(table->get("weight"));
    std::shared_ptr<Tensor> biasPtr = std::dynamic_pointer_cast<Tensor>(table->get("bias"));
    setFilter(filterPtr, biasPtr);

    _padW = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("padW"))->value();
    _padH = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("padH"))->value();
    _dW = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("dW"))->value();
    _dH = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("dH"))->value();
    if (table->get("dilationW") != nullptr)
      _dilationW = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("dilationW"))->value();
    if (table->get("dilationH") != nullptr)
      _dilationH = (int)std::dynamic_pointer_cast<TorchNumber>(table->get("dilationH"))->value();
    return std::shared_ptr<TorchObject>(this);
  }
}
