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

  std::shared_ptr<Sequential> net = std::dynamic_pointer_cast<Sequential>(TorchLoader::getInstance()->loadFile("../net.t7"));
  std::cout << "Done loading net" << std::endl;
  std::shared_ptr<Tensor> image = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../input.t7"));
  if (net == nullptr || image == nullptr)
    return -1;

  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(image);
  std::shared_ptr<Tensor> out = std::dynamic_pointer_cast<Tensor>(net->forward(input));
  std::cout << "Output : " << *out << std::endl;
  auto res = out->read();
  return 0;
}
