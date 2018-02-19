#include "TorchNumber.h"

namespace NN
{
  TorchNumber::TorchNumber()
  {
    _type = TorchType::TorchNumberType;
  }

  std::shared_ptr<TorchObject> TorchNumber::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    _value = std::atof(readNextLine(file).c_str());
    return std::shared_ptr<TorchObject>(this);
  }

  float TorchNumber::value() const
  {
    return _value;
  }
}
