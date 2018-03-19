#include <iostream>

#include "TorchString.h"

namespace NN
{
  TorchString::TorchString()
  {
    _type = TorchType::TorchStringType;
  }

  std::shared_ptr<TorchObject> TorchString::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int nbChars = std::atoi(readNextLine(file).c_str());
    _value = readNextLine(file);
    return std::shared_ptr<TorchObject>(this);
  }

  std::string TorchString::value() const
  {
    return _value;
  }
}
