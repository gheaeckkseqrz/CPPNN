#include "../Sequential.h"
#include "../TorchLoader.h"

using namespace NN;

TEST_CASE( "VGG", "[TorchVGG]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input.t7"));
  std::shared_ptr<Sequential> vgg = std::dynamic_pointer_cast<Sequential>(TorchLoader::getInstance()->loadFile("../Tests/TestData/vgg.t7"));
  REQUIRE_NOTHROW(vgg->forward(input));

  for (int i(0) ; i < 30 ; ++i)
    {
      std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(vgg->forward(input));
    }
}
