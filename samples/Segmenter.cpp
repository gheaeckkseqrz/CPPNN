#include <iostream>
#include "../MatToTensor.h"
#include "Segmenter.h"
#include "TorchLoader.h"

#include "OpenCLFuncs.h"

using namespace NN;

#define SEGMENTER_NETWORK_PATH "../tests/TestData/dilatedVGGReflectionPadding.t7"

int main(int ac, char **av)
{
  std::cout << "Semantic texture segmentation" << std::endl;
  if (ac < 2)
    {
      std::cerr << "Usage : " << av[0] << " INPUT_IMAGE" << std::endl;
      return 1;
    }

  for (int i(1) ; i < ac ; ++i)
    {
      std::shared_ptr<Tensor> t = tensorFromImage(av[i], 512);
      std::cout << "Input : " << *t << std::endl;
      t->sub(127); // -- Preprocessing image for VGG
      Segmenter s(SEGMENTER_NETWORK_PATH);
      std::shared_ptr<Tensor> mask = s.createRGBMask(t);
      saveTensorAsImage(mask, "./mask" + std::to_string(i) + ".png");
    }
  return 0;
}
