#include <sstream>

#include "OpenCL.h"
#include "TorchStorage.h"

namespace NN
{
  TorchStorage::TorchStorage()
    :Storage(0)
  {
  }

  std::shared_ptr<TorchObject> TorchStorage::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    _size = readNextInt(file);
    std::string line = readNextLine(file);
    std::stringstream ss(line);
    std::vector<float> values(_size);
    float v;
    for (int i(0) ; i < _size ; ++i)
      {
	ss >> v;
	values[i] = v;
      }
    _buffer = OpenCL::getInstance()->toGPU<float>(values);
    return std::shared_ptr<TorchObject>(this);
  }
}
