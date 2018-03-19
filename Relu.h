#ifndef __RELU_H__
#define __RELU_H__

#include "Module.h"

namespace NN
{
  class Relu : public Module
  {
  public:
    Relu();

    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input);
    virtual std::string print() const;
  };
}

#endif /* __RELU_H__ */
