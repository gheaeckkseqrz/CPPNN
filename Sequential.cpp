#include <iostream>

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
}
