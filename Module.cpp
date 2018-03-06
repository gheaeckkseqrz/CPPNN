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
}
