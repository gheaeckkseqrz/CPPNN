#ifndef __TANH_H__
#define __TANH_H__

#include "Module.h"

namespace NN
{
  class Tanh : public Module
  {
  public:
    Tanh();

    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input);
    virtual std::string print() const;
  };
}

#endif /* __TANH_H__ */
