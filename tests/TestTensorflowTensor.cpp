#include "TensorflowTensor.h"

using namespace NN;

TEST_CASE( "ParseFloat", "[TensorflowTensor]" )
{
  TensorflowTensor tensor;
  std::vector<std::pair<float, std::string>> tests;
  tests.push_back(std::pair<float, std::string>(0.0f, "\\000\\000\\000\\000"));
  tests.push_back(std::pair<float, std::string>(1.0f, "\\000\\000\\200?"));
  tests.push_back(std::pair<float, std::string>(2.0f, "\\000\\000\\000@"));
  tests.push_back(std::pair<float, std::string>(3.0f, "\\000\\000@@"));
  tests.push_back(std::pair<float, std::string>(4.0f, "\\000\\000\\200@"));
  tests.push_back(std::pair<float, std::string>(5.0f, "\\000\\000\\240@"));
  tests.push_back(std::pair<float, std::string>(6.0f, "\\000\\000\\300@"));
  tests.push_back(std::pair<float, std::string>(7.0f, "\\000\\000\\340@"));
  tests.push_back(std::pair<float, std::string>(8.0f, "\\000\\000\\000A"));
  tests.push_back(std::pair<float, std::string>(9.0f, "\\000\\000\\020A"));
  tests.push_back(std::pair<float, std::string>(10.0f, "\\000\\000\\040A"));

  for (auto t : tests)
    {
      int index = 0;
      REQUIRE(t.first == tensor.parseFloat(t.second, index));
      REQUIRE(index == t.second.size());
    }
}

