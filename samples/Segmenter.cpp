#include <iostream>
#include "../MatToTensor.h"

using namespace NN;

int main(int ac, char **av)
{
  std::cout << "Semantic texture segmentation" << std::endl;
  if (ac < 2)
    {
      std::cerr << "Usage : " << av[0] << " INPUT_IMAGE" << std::endl;
      return 1;
    }

  std::shared_ptr<Tensor> t = tensorFromImage(av[1]);
  

  return 0;
}
