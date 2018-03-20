#include <iostream>
#include "../MatToTensor.h"
#include "Segmenter.h"
#include "TorchLoader.h"

using namespace NN;

#define SEGMENTER_NETWORK_PATH "/Users/wilmot_p/PROG/CPPNN2/Tests/TestData/dilatedVGGReflectionPadding.t7"

int main(int ac, char **av)
{
  std::cout << "Semantic texture segmentation" << std::endl;
  if (ac < 2)
    {
      std::cerr << "Usage : " << av[0] << " INPUT_IMAGE" << std::endl;
      return 1;
    }

  std::shared_ptr<Tensor> t = tensorFromImage(av[1]);
  t->sub(127); // -- Preprocessing image for VGG
  Segmenter s(SEGMENTER_NETWORK_PATH);
  std::shared_ptr<Tensor> mask = s.createRGBMask(t);
  saveTensorAsImage(mask, "./mask.png");
  return 0;
}
