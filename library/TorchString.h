#ifndef __TORCHSTRING_H__
#define __TORCHSTRING_H__

#include "TorchObject.h"

namespace NN
{
  class TorchString : public TorchObject
  {
  public:
    TorchString();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
    std::string value() const;

  private:
    std::string _value;
  };
}

#endif /* __TORCHSTRING_H__ */
