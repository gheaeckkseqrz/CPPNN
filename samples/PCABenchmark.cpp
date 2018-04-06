#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "PCA.h"
#include "Tensor.h"

void reduce(std::shared_ptr<NN::Tensor> data, int targetDim)
{
  NN::PCA p;
  std::cout << "Input : " << *data << std::endl;
  clock_t begin = clock();
  std::shared_ptr<NN::Tensor> result = p.reduceDataToNbOfDims(data, targetDim);
  clock_t end = clock();
  std::cout << "Output : " << *result << std::endl;
  std::cout << "Runtime : " << double(end - begin) / CLOCKS_PER_SEC << " sec" << std::endl;
  std::cout << "=================================================================" << std::endl;
}

int main(int ac, char **av)
{
  std::cout << "PCA Benchmark\n" << std::endl;
  std::vector<int> size({10, 100, 1000, 10000, 100000});
  std::vector<int> depth({2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});

  for (auto s : size)
    {
      for (auto d : depth)
	{
	  std::vector<float> data(s * d);
	  std::generate(data.begin(), data.end(), []() { return rand() % 256; });
	  std::shared_ptr<NN::Tensor> t = std::make_shared<NN::Tensor>(std::vector<int>({d, s}), data);
	  for (auto targetDepth : depth)
	    if (targetDepth < d)
	      reduce(t, targetDepth);
	}
    }


}

