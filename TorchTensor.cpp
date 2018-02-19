#include <sstream>
#include <vector>

#include "TorchLoader.h"
#include "TorchStorage.h"
#include "TorchTensor.h"

namespace NN
{
  TorchTensor::TorchTensor()
    :Tensor()
  {
  }

  std::shared_ptr<TorchObject> TorchTensor::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int nDim = readNextInt(file);
    std::vector<int> sizes;
    if (nDim == 0)
      {
	sizes.push_back(0);
	setSizes(sizes);
	(void)readNextLine(file);
	(void)readNextLine(file);
	return std::shared_ptr<TorchObject>(this);
      }
    std::stringstream dims(readNextLine(file));
    size_t dimSize = 0;
    for (int i(0) ; i < nDim ; ++i)
      {
	dims >> dimSize;
	sizes.push_back(dimSize);
      }
    setSizes(sizes);
    std::string strides = readNextLine(file); // Unused in curent implementation
    if (nDim > 0)
      {
	_offset = readNextInt(file) - 1;
	int storageType = readNextInt(file);
	_storage = std::dynamic_pointer_cast<TorchStorage>(TorchLoader::getInstance()->create(storageType, file, loaded));
      }
    return std::shared_ptr<TorchObject>(this);
  }
}
