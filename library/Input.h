#ifndef __INPUT_H__
#define __INPUT_H__

#include <iostream>

namespace NN
{
  class Input
  {
  public:
    Input() {}
    virtual ~Input() {}

  private:
    /* Input(Input const &o) { std::cout << "Input private copy constructor" << std::endl; } */
  };
}

#endif
