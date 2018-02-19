#ifndef __TORCHNUMBER_H__
#define __TORCHNUMBER_H__

#include "TorchObject.h"

namespace NN
{
  class TorchNumber : public TorchObject
  {
  public:
  public:
    TorchNumber();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
    float value() const;

  private:
    float _value;
  };
}

#endif /* __TORCHNUMBER_H__ */
