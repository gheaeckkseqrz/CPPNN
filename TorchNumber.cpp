#include "TorchNumber.h"

namespace NN
{
  TorchNumber::TorchNumber()
  {
    _type = TorchType::TorchNumberType;
  }

  TorchObject *TorchNumber::loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded)
  {
    _value = std::atof(readNextLine(file).c_str());
    return this;
  }

  float TorchNumber::value() const
  {
    return _value;
  }
}
