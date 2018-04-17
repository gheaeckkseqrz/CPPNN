#include <iostream>

#include "../MatToTensor.h"
#include "TextureLibrary.h"

using namespace NN;

int main(int ac, char **av)
{
  TextureLibrary t;
  t.addDirectory("/home/wilmot_p/DATA2/DATASETS/JOHN_DATABASE");

  std::shared_ptr<Tensor> example = tensorFromImage("/home/wilmot_p/wall.png");
  t.findNN(example, 5);

  return 0;
}
