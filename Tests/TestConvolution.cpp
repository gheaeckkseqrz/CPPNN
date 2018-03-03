#include "../Convolution.h"

using namespace NN;

TEST_CASE( "Convolution empty constructor", "[Convolution]" )
{
  NN::Convolution c;
  std::shared_ptr<NN::Tensor> t = std::make_shared<NN::Tensor>(std::vector<int>({5, 5, 5}));
  REQUIRE_THROWS(c.forward(t));
}

TEST_CASE( "Convolution with filter / input mismatch", "[Convolution]" )
{
  NN::Convolution c;
  std::shared_ptr<NN::Tensor> filter = std::make_shared<NN::Tensor>(std::vector<int>({5, 5, 5, 5}));
  std::shared_ptr<NN::Tensor> t = std::make_shared<NN::Tensor>(std::vector<int>({6, 5, 5}));
  c.setFilter(filter);
  REQUIRE_THROWS(c.forward(t));
}

TEST_CASE( "Convolution with dummy 1 channel filter / input and padding", "[Convolution]" )
{
  NN::Convolution c;
  std::shared_ptr<NN::Tensor> filter = std::make_shared<NN::Tensor>(std::vector<int>({1, 1, 3, 3}));
  std::shared_ptr<NN::Tensor> t = std::make_shared<NN::Tensor>(std::vector<int>({1, 5, 5}));
  filter->fill(1);
  t->fill(1);
  c.setFilter(filter);
  std::shared_ptr<NN::Input> computed;
  REQUIRE_NOTHROW(computed = c.forward(t));
  std::shared_ptr<NN::Tensor> computedTensor = std::dynamic_pointer_cast<NN::Tensor>(computed);;
  std::vector<float> computedValues = computedTensor->read();
  std::vector<float> expectedResult = std::vector<float>({4,6,6,6,4,6,9,9,9,6,6,9,9,9,6,6,9,9,9,6,4,6,6,6,4});
  REQUIRE( computedTensor->getSizes() == std::vector<int>({1, 5, 5}) );
  REQUIRE( computedValues == expectedResult );
}

TEST_CASE( "Convolution with dummy 3/3 channels and padding", "[Convolution]" )
{
  NN::Convolution c;
  std::vector<float> filterValues(3*3*3*3);
  for (int i(0) ; i < 3*3*3*3 ; ++i)
    filterValues[i] = i;
  std::shared_ptr<NN::Tensor> filter = std::make_shared<NN::Tensor>(std::vector<int>({3, 3, 3, 3}), filterValues);
  std::shared_ptr<NN::Tensor> t = std::make_shared<NN::Tensor>(std::vector<int>({3, 5, 5}));
  t->fill(1);
  c.setFilter(filter);
  std::shared_ptr<NN::Input> computed;
  REQUIRE_NOTHROW(computed = c.forward(t));
  std::shared_ptr<NN::Tensor> computedTensor = std::dynamic_pointer_cast<NN::Tensor>(computed);;
  std::vector<float> computedValues = computedTensor->read();
  std::vector<float> expectedResult = std::vector<float>({180,261,261,261,168,243,351,351,351,225,243,351,351,351,225,243,351,351,351,225,144,207,207,207,132,504,747,747,747,492,729,1080,1080,1080,711,729,1080,1080,1080,711,729,1080,1080,1080,711,468,693,693,693,456,828,1233,1233,1233,816,1215,1809,1809,1809,1197,1215,1809,1809,1809,1197,1215,1809,1809,1809,1197,792,1179,1179,1179,780});
  REQUIRE( computedTensor->getSizes() == std::vector<int>({3, 5, 5}) );
  REQUIRE( computedValues == expectedResult );
}

TEST_CASE( "Convolution with dummy 3/3 channels and no padding", "[Convolution]" )
{
  NN::Convolution c;
  std::vector<float> filterValues(3*3*3*3);
  for (int i(0) ; i < 3*3*3*3 ; ++i)
    filterValues[i] = i;
  std::shared_ptr<NN::Tensor> filter = std::make_shared<NN::Tensor>(std::vector<int>({3, 3, 3, 3}), filterValues);
  std::shared_ptr<NN::Tensor> t = std::make_shared<NN::Tensor>(std::vector<int>({3, 5, 5}));
  t->fill(1);
  c.setFilter(filter);
  c.setPadding(0, 0);
  std::shared_ptr<NN::Input> computed;
  REQUIRE_NOTHROW(computed = c.forward(t));
  std::shared_ptr<NN::Tensor> computedTensor = std::dynamic_pointer_cast<NN::Tensor>(computed);;
  std::vector<float> computedValues = computedTensor->read();
  std::vector<float> expectedResult = std::vector<float>({351,351,351,351,351,351,351,351,351,1080,1080,1080,1080,1080,1080,1080,1080,1080,1809,1809,1809,1809,1809,1809,1809,1809,1809});
  REQUIRE( computedTensor->getSizes() == std::vector<int>({3, 3, 3}) );
  REQUIRE( computedValues == expectedResult );
}

