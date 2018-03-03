#ifndef __REFLECTIONPADDING_H__
#define __REFLECTIONPADDING_H__

#include "Module.h"

namespace NN
{
  class ReflectionPadding : public Module
  {
  public:
    ReflectionPadding();

    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input);

  protected:
    int _padB;
    int _padT;
    int _padR;
    int _padL;
  };
}

#endif /* __REFLECTIONPADDING_H__ */
