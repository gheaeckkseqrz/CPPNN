#ifndef __MAXPOOLING_H__
#define __MAXPOOLING_H__

#include "Module.h"

namespace NN
{
  class MaxPooling : public Module
  {
  public:
    MaxPooling();

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> const &input);
    virtual std::string print() const;
  };
}

#endif /* __MAXPOOLING_H__ */
