#include <cstdlib>
#include <ctime>
#include <iostream>
#include "Convolution.h"
#include "Tensor.h"

#define CHANNELS 512

int main(int ac, char **av)
{
  (void)ac;
  (void)av;

  std::shared_ptr<NN::Tensor> filter = std::make_shared<NN::Tensor>(std::vector<int>({CHANNELS, CHANNELS, 3, 3}));
  NN::Convolution c;
  c.setFilter(filter);

  int s = 50;
  while (s <= 1500)
    {
      std::shared_ptr<NN::Tensor> t = std::make_shared<NN::Tensor>(std::vector<int>({CHANNELS, s, s}));
      clock_t begin = clock();
      std::shared_ptr<NN::Tensor> res = c.forward(t);
      clock_t end = clock();
      std::cout << *t << " => " << double(end - begin) / CLOCKS_PER_SEC << " sec" << std::endl;
      s += 50;
    }
  return 0;
}
