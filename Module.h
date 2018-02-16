#ifndef __MODULE_H__
#define __MODULE_H__

#include "Input.h"

namespace NN
{
  class Module
  {
  public:
    Module();

    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input) = 0;

  protected:
    std::shared_ptr<Input> _output;
  };
}

#endif /* __MODULE_H__ */
