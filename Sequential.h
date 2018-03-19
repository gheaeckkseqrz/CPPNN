#ifndef __SEQUENTIAL_H__
#define __SEQUENTIAL_H__

#include <vector>

#include "Module.h"

namespace NN
{
  class Sequential : public Module
  {
  public:
    Sequential();

    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input);
    std::shared_ptr<Module> get(unsigned int index) const;
    virtual std::string print() const;

 protected:
    std::vector<std::shared_ptr<Module>> _modules;
  };
}

#endif /* __SEQUENTIAL_H__ */
