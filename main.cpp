#include <iostream>
#include <unistd.h>

#include "Sequential.h"
#include "Tensor.h"
#include "TorchLoader.h"
#include "TorchSequential.h"

using namespace NN;

int main(int ac, char **av)
{
  (void)ac;
  (void)av;
  std::cout << "Hello World" << std::endl;
  Sequential *net = dynamic_cast<TorchSequential*>(TorchLoader::getInstance()->loadFile("../net.t7"));
  Tensor *image = dynamic_cast<Tensor*>(TorchLoader::getInstance()->loadFile("../input.t7"));
  std::shared_ptr<Input> input(image);
  Tensor *out = dynamic_cast<Tensor*>(net->forward(input).get());
  std::cout << "Output : " << *out << std::endl;
  auto res = out->read();
  return 0;
}
