#include <cassert>
#include <iostream>
#include <stdexcept>

#include "Sequential.h"

namespace NN
{
  Sequential::Sequential()
  {
  }

  std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> const &input)
  {
    std::shared_ptr<Tensor> current = input;
    for (int i(0) ; i < _modules.size() ; ++i)
      {
	//	std::cout << "[Sequential " << i + 1 << " / " << _modules.size() << "] - " << *(_modules[i]) << std::endl;
    	current = _modules[i]->forward(current);
	if (!_outputsRetainPolicy.empty() && _outputsRetainPolicy[i] == false)
	  _modules[i]->clearOutput();
      }
    return current;
  }

  std::shared_ptr<Module> Sequential::get(unsigned int index) const
  {
    if (index >= _modules.size())
      throw std::runtime_error("Invalid index for Sequential::get()");
    return _modules[index];
  }

  void Sequential::remove(unsigned int index)
  {
    // std::cout << "Removing " << *_modules[index] << std::endl;
    _modules.erase(_modules.begin() + index);
  }

  size_t Sequential::size() const
  {
    return _modules.size();
  }

  void Sequential::setRetainPolicy(std::vector<bool> const &policy)
  {
    assert(_modules.size() == policy.size());
    _outputsRetainPolicy = policy;
  }

  std::string Sequential::print() const
  {
    return "Sequential";
  }
}
