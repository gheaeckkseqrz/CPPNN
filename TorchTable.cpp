#include <iostream>

#include "TorchLoader.h"
#include "TorchNumber.h"
#include "TorchString.h"
#include "TorchTable.h"

namespace NN
{
  TorchTable::TorchTable()
  {
    _type = TorchType::TorchTableType;
  }

  TorchTable::~TorchTable()
  {
    std::cout << "TorchTable destructor " << _name << std::endl;
  }

  TorchObject *TorchTable::loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded)
  {
    int id = std::atoi(readNextLine(file).c_str());
    int tableSize = std::atoi(readNextLine(file).c_str());
    for (int i = 1; i <= tableSize; ++i)
      {
	int keyType = std::atoi(readNextLine(file).c_str());
	TorchObject *k = TorchLoader::getInstance()->create(keyType, file, loaded);
	int valueType = std::atoi(readNextLine(file).c_str());
	TorchObject *v = TorchLoader::getInstance()->create(valueType, file, loaded);
	if (k->getType() == TorchObject::TorchType::TorchNumberType)
	  _intKeys[(int)((TorchNumber*)k)->value()] = v;
	else if (k->getType() == TorchObject::TorchType::TorchStringType)
	  _stringKeys[((TorchString*)k)->value()] = v;
	else
	  std::cerr << "Unknow object type " << k->getType() << std::endl;
      }
    return this;
  }

  TorchObject *TorchTable::get(int key) const
  {
    if (_intKeys.count(key) == 0)
      return nullptr;
    return _intKeys.at(key);
  }

  TorchObject *TorchTable::get(std::string const &key) const
  {
    if (_stringKeys.count(key) == 0)
      return nullptr;
    return _stringKeys.at(key);
  }
}
