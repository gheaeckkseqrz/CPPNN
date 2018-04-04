#include <ctime>
#include <iostream>
#include "../MatToTensor.h"
#include "Segmenter.h"

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

  clock_t begin = clock();
  Segmenter s(SEGMENTER_NETWORK_PATH);
  clock_t end = clock();
  std::cout << "Loading network " << double(end - begin) / CLOCKS_PER_SEC << " sec" << std::endl;

  double runtime = 0;
  for (int i(1) ; i < ac ; ++i)
    {
      std::shared_ptr<Tensor> t = tensorFromImage(av[i], 512);
      saveTensorAsImage(t, "./input_" + std::to_string(i) + ".png");
      t->sub(127); // -- Preprocessing image for VGG
      clock_t begin = clock();
      std::shared_ptr<Tensor> mask = s.createRGBMask(t);
      clock_t end = clock();
      saveTensorAsImage(mask, "./PCA_mask_" + std::to_string(i) + ".png");
      std::cout << "Processed " << av[i] << " Runtime : " << double(end - begin) / CLOCKS_PER_SEC << " sec" << std::endl;
      runtime += double(end - begin);
    }
  std::cout << "Average runtime : " << runtime / CLOCKS_PER_SEC / (ac - 1) << " sec" << std::endl;
  return 0;
}
