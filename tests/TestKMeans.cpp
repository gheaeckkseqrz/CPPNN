#include "Kmeans.h"

TEST_CASE( "1D - Centroid init" "[KMeans]" )
{
  std::vector<float> points({0, 4, 5});
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<int>({1, 3}), points);
  Kmeans o(2);
  o.clusterData(t, 0);
  std::vector<float> expectedCentroids({3, 0});
  REQUIRE(o.getCentroids()->read() == expectedCentroids);
}

TEST_CASE( "1D - Kmeans" "[KMeans]" )
{
  std::vector<float> points({0, 4, 5});
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<int>({1, 3}), points);
  Kmeans o(2);
  o.clusterData(t, 1);
  std::vector<float> expectedResult({1, 0, 0});
  REQUIRE(o.clusterData(t)->read() == expectedResult);
}

TEST_CASE( "1D - Update Centroids" "[KMeans]" )
{
  std::vector<float> points({0, 4, 5});
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<int>({1, 3}), points);
  Kmeans o(2);
  o.clusterData(t, 1);
  std::vector<float> expectedCentroids({4.5, 0});
  REQUIRE(o.getCentroids()->read() == expectedCentroids);
}

TEST_CASE( "2D - Centroid init" "[KMeans]" )
{
  std::vector<float> points({1, -7, 12, -5, 8, 2, -3, -7});
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<int>({2, 4}), points);
  Kmeans o(2);
  o.clusterData(t, 0);
  std::vector<float> expectedCentroids({0.25, 0, 12, -3});
  REQUIRE(o.getCentroids()->read() == expectedCentroids);
}

TEST_CASE( "2D - Kmeans" "[KMeans]" )
{
  std::vector<float> points({1, -7, 12, -5, 8, 2, -3, -7});
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<int>({2, 4}), points);
  Kmeans o(2);
  o.clusterData(t, 1);
  std::vector<float> expectedResult({0, 0, 1, 0});
  REQUIRE(o.clusterData(t)->read() == expectedResult);
}

TEST_CASE( "2D - Update Centroids" "[KMeans]" )
{
  std::vector<float> points({1, -7, 12, -5, 8, 2, -3, -7});
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<int>({2, 4}), points);
  Kmeans o(2);
  o.clusterData(t, 1);
  std::vector<float> expectedCentroids({-11.0f/3, 3/3, 12, -3});
  REQUIRE(o.getCentroids()->read() == expectedCentroids);
}
