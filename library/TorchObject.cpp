#include <iostream>
#include <string>

#include "TorchObject.h"

namespace NN
{
  TorchObject::TorchObject()
  {
    _type = TorchType::TorchNilType;
  }

  TorchObject::~TorchObject()
  {
  }

  TorchObject::TorchType TorchObject::getType() const
  {
    return _type;
  }

  int readNextInt(std::ifstream &file)
  {
    std::string line = readNextLine(file);
    return std::atoi(line.c_str());
  }

  std::string readNextLine(std::ifstream &file)
  {
    std::string line;
    std::getline(file, line);
    return line;
  }
}
