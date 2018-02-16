#include "TorchLoader.h"
#include "TorchSequential.h"
#include "TorchTable.h"

namespace NN
{
  TorchSequential::TorchSequential()
  {
  }

  TorchObject *TorchSequential::loadFromFile(std::ifstream &file, std::map<int, TorchObject *> &loaded)
  {
    int tableId = readNextInt(file);
    TorchTable *t = (TorchTable*)TorchLoader::getInstance()->create(tableId, file, loaded);
    TorchTable *modules = (TorchTable*)t->get("modules");
    int i = 1;
    while (modules->get(i) != nullptr)
      {
	_modules.push_back(dynamic_cast<Module*>(modules->get(i)));
	i++;
      }
    return this;
  }
}
