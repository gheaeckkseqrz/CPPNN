#ifndef __MAXPOOLING_H__
#define __MAXPOOLING_H__

#include "Module.h"

namespace NN
{
  class MaxPooling : public Module
  {
  public:
    MaxPooling();

    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input);
  };
}

#endif /* __MAXPOOLING_H__ */
