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

  TorchObject *TorchConvolution::loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded)
  {
    int tableId = readNextInt(file);
    TorchTable *t = (TorchTable*)TorchLoader::getInstance()->create(tableId, file, loaded);

    std::shared_ptr<Tensor> filterPtr;
    filterPtr.reset((TorchTensor*)t->get("weight"));
    std::shared_ptr<Tensor> biasPtr;
    biasPtr.reset((TorchTensor*)t->get("bias"));

    setFilter(filterPtr, biasPtr);

    _padW = (int)((TorchNumber*)t->get("padW"))->value();
    _padH = (int)((TorchNumber*)t->get("padH"))->value();
    _dW = (int)((TorchNumber*)t->get("dW"))->value();
    _dH = (int)((TorchNumber*)t->get("dH"))->value();
    return this;
  }
}
