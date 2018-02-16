#ifndef __TORCHTABLE_H__
#define __TORCHTABLE_H__

#include "TorchObject.h"

namespace NN
{
  class TorchTable : public TorchObject
  {
  public:
    TorchTable();
    virtual ~TorchTable();

    virtual TorchObject *loadFromFile(std::ifstream &file, std::map<int, TorchObject*> &loaded);

    TorchObject *get(int key) const;
    TorchObject *get(std::string const &key) const;

  private:
    std::map<int, TorchObject*>         _intKeys;
    std::map<std::string, TorchObject*> _stringKeys;
  };
}

#endif /* __TORCHTABLE_H__ */
