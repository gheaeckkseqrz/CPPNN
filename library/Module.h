#ifndef __MODULE_H__
#define __MODULE_H__

#include <memory>
#include "Tensor.h"

namespace NN
{
  class Module
  {
  public:
    Module();
    virtual ~Module() {}

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> const &input) = 0;
    virtual std::string print() const = 0;
    std::shared_ptr<Tensor> getOutput() const;
    void clearOutput();

  protected:
    std::shared_ptr<Tensor> _output;
  };

  std::ostream &operator<<(std::ostream &s, Module const &m);
}


#endif /* __MODULE_H__ */
