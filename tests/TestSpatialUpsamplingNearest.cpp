#include "SpatialUpsamplingNearest.h"

using namespace NN;

TEST_CASE( "SpatialUpsamplingNearest", "[SpatialUpsamplingNearest]" )
{
  NN::SpatialUpsamplingNearest m;
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<int>({2, 2, 2}), std::vector<float>(8, 1));
  std::shared_ptr<Tensor> output;
  REQUIRE_NOTHROW( output = m.forward(input) );
  std::vector<float> outputData = output->read();
  REQUIRE( output->getSizes() == std::vector<int>({2, 4, 4}) );
  REQUIRE( outputData == std::vector<float>(32, 1) );
}

TEST_CASE( "SpatialUpsamplingNearest2", "[SpatialUpsamplingNearest]" )
{
  NN::SpatialUpsamplingNearest m;
  std::vector<float> data({1, 2, 3, 4, 5, 6, 7, 8});
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<int>({2, 2, 2}), data);
  std::shared_ptr<Tensor> output;
  REQUIRE_NOTHROW( output = m.forward(input) );
  std::vector<float> outputData = output->read();
  REQUIRE( output->getSizes() == std::vector<int>({2, 4, 4}) );
  REQUIRE( outputData == std::vector<float>({1, 1, 2, 2,
	  1, 1, 2, 2,
	  3, 3, 4, 4,
	  3, 3, 4, 4,
	  5, 5, 6, 6,
	  5, 5, 6, 6,
	  7, 7, 8, 8,
	  7, 7, 8, 8}) );
}

TEST_CASE( "SpatialUpsamplingNearest bad input shape",  "[SpatialUpsamplingNearest]" )
{
  NN::SpatialUpsamplingNearest m;
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<int>({2, 2}), std::vector<float>(4, 1));
  REQUIRE_THROWS( m.forward(input) );
}
