#include <iostream>

#include "TorchString.h"

namespace NN
{
  TorchString::TorchString()
  {
    _type = TorchType::TorchStringType;
  }

  TorchObject *TorchString::loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded)
  {
    int nbChars = std::atoi(readNextLine(file).c_str());
    _value = readNextLine(file);
    return this;
  }

  std::string TorchString::value() const
  {
    return _value;
  }
}
