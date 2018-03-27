#ifndef __REFLECTIONPADDING_H__
#define __REFLECTIONPADDING_H__

#include "Module.h"

namespace NN
{
  class ReflectionPadding : public Module
  {
  public:
    ReflectionPadding();

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> const &input);
    virtual std::string print() const;

  protected:
    int _padB;
    int _padT;
    int _padR;
    int _padL;
  };
}

#endif /* __REFLECTIONPADDING_H__ */
