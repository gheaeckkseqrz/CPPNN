#include "TorchLoader.h"
#include "TorchSequential.h"
#include "TorchTable.h"

namespace NN
{
  TorchSequential::TorchSequential()
  {
  }

  std::shared_ptr<TorchObject> TorchSequential::loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded)
  {
    int tableId = readNextInt(file);
    std::shared_ptr<TorchTable> t = std::dynamic_pointer_cast<TorchTable>(TorchLoader::getInstance()->create(tableId, file, loaded));
    std::shared_ptr<TorchTable> modules = std::dynamic_pointer_cast<TorchTable>(t->get("modules"));
    int i = 1;
    while (modules->get(i) != nullptr)
      {
	_modules.push_back(std::dynamic_pointer_cast<Module>(modules->get(i)));
	i++;
      }
    return std::shared_ptr<TorchObject>(this);
  }
}
