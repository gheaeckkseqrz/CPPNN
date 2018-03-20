#include "Module.h"

namespace NN
{
  Module::Module()
  {
  }

  std::shared_ptr<Input> Module::getOutput() const
  {
    return _output;
  }

  void Module::clearOutput()
  {
    _output = nullptr;
  }

  std::ostream &operator<<(std::ostream &s, Module const &m)
  {
    s << m.print();
    return s;
  }
}
