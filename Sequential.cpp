#include <iostream>
#include <stdexcept>

#include "Sequential.h"

namespace NN
{
  Sequential::Sequential()
  {
  }

  std::shared_ptr<Input> Sequential::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Input> current = input;
    for (int i(0) ; i < _modules.size() ; ++i)
      {
    	std::cout << "[Sequential " << i << " / " << _modules.size() << "]" << std::endl;
    	current = _modules[i]->forward(current);
      }
    return current;
  }

  std::shared_ptr<Module> Sequential::get(unsigned int index) const
  {
    if (index >= _modules.size())
      throw std::runtime_error("Invalid index for Sequential::get()");
    return _modules[index];
  }
}
