#include <iostream>

#include "../MatToTensor.h"
#include "TextureLibrary.h"

using namespace NN;

int main(int ac, char **av)
{
  TextureLibrary t;
  t.addDirectory("/home/wilmot_p/DATA2/DATASETS/JOHN_DATABASE");
  //  t.addImage("/home/wilmot_p/Pictures/GROS/2017_0724_011333_001.JPG");
  return 0;
}
