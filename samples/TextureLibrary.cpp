#include <ctime>
#include <iostream>

#include "../MatToTensor.h"
#include "TextureLibrary.h"

using namespace NN;

int main(int ac, char **av)
{
  TextureLibrary t;
  clock_t begin = clock();
  std::cout << "Building library ..." << std::endl;
  t.addDirectory("/home/wilmot_p/DATA2/DATASETS/JOHN_DATABASE");
  clock_t end = clock();
  std::cout << "Building library took " << double(end - begin) / CLOCKS_PER_SEC / 3600 << " hours" << std::endl;

  while (true)
    {
      std::string path;
      std::cout << ">> ";
      std::cin >> path;
      if (path == "exit")
	break;
      std::shared_ptr<Tensor> example = tensorFromImage(path, 512);
      if (example)
	{
	  clock_t begin = clock();
	  std::cout << "Input : " << path << std::endl;
	  std::vector<std::string> ret = t.findNN(example, 5, (ParametricModel::e_relu2_1 | ParametricModel::e_relu3_1 | ParametricModel::e_relu4_1 | ParametricModel::e_relu5_1));
	  for (std::string result : ret)
	    std::cout << result << std::endl;
	  clock_t end = clock();
	  std::cout << "Search " << double(end - begin) / CLOCKS_PER_SEC << " seconds" << std::endl;
	}

    }



  return 0;
}
