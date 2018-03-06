#include "../TorchConvolution.h"
#include "../TorchLoader.h"

using namespace NN;

TEST_CASE( "TorchConvolution 1 In 1 Out No Padding No Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outNoPaddingNoBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outNoPaddingNoBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));
  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution 1 In 1 Out Padding No Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outPaddingNoBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outPaddingNoBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution 1 In 1 Out No Padding Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outNoPaddingBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outNoPaddingBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution 1 In 1 Out Padding Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outPaddingBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolution1in1outPaddingBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In < Out No Padding No Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutNoPaddingNoBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutNoPaddingNoBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In < Out Padding No Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutPaddingNoBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutPaddingNoBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In > Out No Padding No Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutNoPaddingNoBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutNoPaddingNoBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In > Out Padding No Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutPaddingNoBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutPaddingNoBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In < Out No Padding Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutNoPaddingBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutNoPaddingBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In < Out Padding Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input1Channel.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutPaddingBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInLTOutPaddingBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In > Out No Padding Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutNoPaddingBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutNoPaddingBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}

TEST_CASE( "TorchConvolution In > Out Padding Bias", "[TorchConvolution]" )
{
  std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(TorchLoader::getInstance()->loadFile("../Tests/TestData/input.t7"));
  std::shared_ptr<Tensor> expected = std::dynamic_pointer_cast<Tensor>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutPaddingBiasResult.t7"));
  std::shared_ptr<Convolution> conv = std::dynamic_pointer_cast<Convolution>(TorchLoader::getInstance()->loadFile("../Tests/TestData/convolutionInGTOutPaddingBias.t7"));
  std::shared_ptr<Tensor> result = std::dynamic_pointer_cast<Tensor>(conv->forward(input));

  REQUIRE(expected->getSizes() == result->getSizes());
  REQUIRE(result->dataEquals(*(expected.get()), 0.0001));
}
