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

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);

    std::shared_ptr<TorchObject> get(int key) const;
    std::shared_ptr<TorchObject> get(std::string const &key) const;

  private:
    std::map<int, std::shared_ptr<TorchObject>>         _intKeys;
    std::map<std::string, std::shared_ptr<TorchObject>> _stringKeys;
  };
}

#endif /* __TORCHTABLE_H__ */
