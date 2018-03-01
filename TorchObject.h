#ifndef __TORCHOBJECT_H__
#define __TORCHOBJECT_H__

#include <map>
#include <memory>
#include <fstream>

namespace NN
{
  class TorchObject
  {
  public:
    enum TorchType
    {
    TorchNilType = 0,
    TorchNumberType = 1,
    TorchStringType = 2,
    TorchTableType = 3,
    TorchObjectType = 4,
    TorchBooleanType = 5
    };

  public:
    TorchObject();
    virtual ~TorchObject();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded) = 0;
    TorchType getType() const;

    std::string _name;

  protected:
    TorchType _type;
  };

  int readNextInt(std::ifstream &file);
  std::string readNextLine(std::ifstream &file);
}

#endif /* __TORCHOBJECT_H__ */
