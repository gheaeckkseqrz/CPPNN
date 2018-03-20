#ifndef __MODULE_H__
#define __MODULE_H__

#include <memory>
#include "Input.h"

namespace NN
{
  class Module
  {
  public:
    Module();

    virtual std::shared_ptr<Input> forward(std::shared_ptr<Input> const &input) = 0;
    virtual std::string print() const = 0;
    std::shared_ptr<Input> getOutput() const;
    void clearOutput();

  protected:
    std::shared_ptr<Input> _output;
  };

  std::ostream &operator<<(std::ostream &s, Module const &m);
}


#endif /* __MODULE_H__ */
