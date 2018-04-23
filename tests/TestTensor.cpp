#include "Tensor.h"

using namespace NN;

TEST_CASE( "Tensor constructor from size vector", "[Tensor]" )
{
  std::vector<int> s({5, 6, 7});
  Tensor t(s);

  REQUIRE( t.getNbElements() == 210 );
  REQUIRE( t.read() == std::vector<float>(210, 0) );
  REQUIRE( t.getSizes() == s );
  REQUIRE( t.getSize(0) == 5 );
  REQUIRE( t.getSize(1) == 6 );
  REQUIRE( t.getSize(2) == 7 );
  REQUIRE( t.getOffset() == 0 );
}

TEST_CASE( "Tensor constructor from data", "[Tensor]" )
{
  std::vector<float> v({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  Tensor t(v);

  REQUIRE( t.getNbElements() == 10 );
  REQUIRE( t.read() == v );
  REQUIRE( t.getSizes() == std::vector<int>({10}) );
  REQUIRE( t.getSize(0) == 10 );
  REQUIRE( t.getOffset() == 0 );
}

TEST_CASE( "Tensor change size", "[Tensor]")
{
  std::vector<int> s({5, 6, 7});
  Tensor t(s);

  REQUIRE( t.getNbElements() == 210 );
  REQUIRE( t.read() == std::vector<float>(210, 0) );
  REQUIRE( t.getSizes() == s );
  REQUIRE( t.getSize(0) == 5 );
  REQUIRE( t.getSize(1) == 6 );
  REQUIRE( t.getSize(2) == 7 );
  REQUIRE( t.getOffset() == 0 );
  t.setSizes(std::vector<int>({7, 5, 6}));
  REQUIRE( t.getNbElements() == 210 );
  REQUIRE( t.read() == std::vector<float>(210, 0) );
  REQUIRE( t.getSizes() == std::vector<int>({7, 5, 6}));
  REQUIRE( t.getSize(0) == 7 );
  REQUIRE( t.getSize(1) == 5 );
  REQUIRE( t.getSize(2) == 6 );
  REQUIRE( t.getOffset() == 0 );
}

TEST_CASE( "Tensor Fill", "[Tensor]" )
{
  std::vector<int> s({5, 6, 7});
  Tensor t(s);

  t.fill(1410.1991);
  std::vector<float> computed = t.read();
  REQUIRE( computed.size() == 210 );
  for (int i(0) ; i < computed.size() ; ++i)
    REQUIRE( Approx(computed[i]) == 1410.1991 );
  t.fill(42);
  computed = t.read();
  REQUIRE( computed.size() == 210 );
  for (int i(0) ; i < computed.size() ; ++i)
    REQUIRE( Approx(computed[i]) == 42 );
  t.fill(-8888);
  computed = t.read();
  REQUIRE( computed.size() == 210 );
  for (int i(0) ; i < computed.size() ; ++i)
    REQUIRE( Approx(computed[i]) == -8888 );
}

TEST_CASE( "Tensor Copy", "[Tensor]" )
{
  Tensor t(std::vector<int>({10}));
  std::vector<float> data({1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
  t.copy(data);
  REQUIRE(t.read() == data);
}

TEST_CASE( "Tensor Means", "[Tensor]" )
{
  std::vector<float> v({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5});
  Tensor t(std::vector<int>({5, 3}), v);
  Tensor m = t.means();
  REQUIRE(m.getSizes() == std::vector<int>({5}));
  REQUIRE(m.read() == std::vector<float>({1, 2, 3, 4, 5}));
}

TEST_CASE( "Tensor Data Equals", "[Tensor]" )
{
  std::vector<float>     v1({1 , 2, 3, 4, 5, 6, 7 , 8   , 9   , 10, -5 , 0   });
  std::vector<float>     v2({9 , 6, 2, 5, 1, 0, 5 , 4.7 , 9.8 , 12, 7.3, -4.1});
  Tensor t1(v1);
  Tensor t2(v2);
  REQUIRE(t1.dataEquals(t1));
  REQUIRE(!t1.dataEquals(t2));
  REQUIRE(!t2.dataEquals(t1));
  REQUIRE(t2.dataEquals(t2));
}

TEST_CASE( "Tensor Element wise addition", "[Tensor]" )
{
  std::vector<float>     v1({1 , 2, 3, 4, 5, 6, 7 , 8   , 9   , 10, -5 , 0   });
  std::vector<float>     v2({9 , 6, 2, 5, 1, 0, 5 , 4.7 , 9.8 , 12, 7.3, -4.1});
  std::vector<float> result({10, 8, 5, 9, 6, 6, 12, 12.7, 18.8, 22, 2.3, -4.1});
  Tensor t1(v1);
  Tensor t2(v2);

  REQUIRE ( t1.read() == v1 );
  REQUIRE ( t2.read() == v2 );
  t1.add(t2);
  REQUIRE ( t2.read() == v2 );
  std::vector<float> computed = t1.read();
  for (int i(0) ; i < computed.size() ; ++i)
    REQUIRE( Approx(computed[i]) == result[i] );
}

TEST_CASE( "Tensor Element wise addition with offset", "[Tensor]" )
{
  std::vector<float>                  v1({1, 2, 3, 4,   5,    6,  7,    8});
  std::vector<float>     v2({9 , 6, 2, 5, 1, 0, 5, 4.7, 9.8,  12, 7.3,  -4.1});
  std::vector<float>              result({2, 2, 8, 8.7, 14.8, 18, 14.3, 3.9});
  Tensor t1(v1);
  Tensor t2(v2);
  t2.setOffset(4);

  REQUIRE ( t1.read() == v1 );
  REQUIRE ( t2.read() == std::vector<float>({1, 0, 5, 4.7, 9.8,  12, 7.3,  -4.1}) );
  REQUIRE_NOTHROW(t1.add(t2));
  std::vector<float> computed = t1.read();
  for (int i(0) ; i < computed.size() ; ++i)
    REQUIRE( Approx(computed[i]) == result[i] );
}

TEST_CASE( "Tensor Element wise substraction", "[Tensor]" )
{
  std::vector<float>     v1({1 , 2,  3, 4,  5, 6, 7 , 8,   9,    10, -5 ,   0   });
  std::vector<float>     v2({9 , 6,  2, 5,  1, 0, 5 , 4.7, 9.8,  12, 7.3,   -4.1});
  std::vector<float> result({-8, -4, 1, -1, 4, 6, 2,  3.3, -0.8, -2, -12.3, 4.1});
  Tensor t1(v1);
  Tensor t2(v2);

  REQUIRE ( t1.read() == v1 );
  REQUIRE ( t2.read() == v2 );
  REQUIRE_NOTHROW(t1.sub(t2));
  REQUIRE ( t2.read() == v2 );
  std::vector<float> computed = t1.read();
  for (int i(0) ; i < computed.size() ; ++i)
    REQUIRE( Approx(computed[i]) == result[i] );
}

TEST_CASE( "Tensor Element wise substraction with offset", "[Tensor]" )
{
  std::vector<float>                  v1({1, 2, 3,  4,    5,    6,  7,    8});
  std::vector<float>     v2({9 , 6, 2, 5, 1, 0, 5,  4.7,  9.8,  12, 7.3,  -4.1});
  std::vector<float>              result({0, 2, -2, -0.7, -4.8, -6, -0.3, 12.1});
  Tensor t1(v1);
  Tensor t2(v2);
  t2.setOffset(4);

  REQUIRE ( t1.read() == v1 );
  REQUIRE ( t2.read() == std::vector<float>({1, 0, 5, 4.7, 9.8, 12, 7.3, -4.1}) );
  REQUIRE_NOTHROW(t1.sub(t2));
  REQUIRE ( t2.read() == std::vector<float>({1, 0, 5, 4.7, 9.8, 12, 7.3, -4.1}) );
  std::vector<float> computed = t1.read();
  for (int i(0) ; i < computed.size() ; ++i)
    REQUIRE( Approx(computed[i]) == result[i] );
}

TEST_CASE( "Tensor Covariance - Square Tensor", "[Tensor]" )
{
  std::vector<float> data;
  for (int i(1) ; i <= 25 ; ++i)
    data.push_back(i);
  Tensor t(std::vector<int>({5, 5}), data);
  std::shared_ptr<Tensor> cov = t.covariance();
  std::vector<float> expectedCovariance(25, 2.5f);
  REQUIRE(cov->read() == expectedCovariance);
}

TEST_CASE( "Tensor Covariance - Non Square Tensor", "[Tensor]" )
{
  std::vector<float> data;
  for (int i(1) ; i <= 30 ; ++i)
    data.push_back(i);
  Tensor t(std::vector<int>({5, 6}), data);
  std::shared_ptr<Tensor> cov = t.covariance();
  std::vector<float> expectedCovariance(25, 3.5);
  REQUIRE(cov->read() == expectedCovariance);
}

TEST_CASE( "Tensor Flat Covariance", "[Tensor]" )
{
  std::vector<float> data({1, 2, 3, 6, 12, 24, 48, 96, 192});
  Tensor t(std::vector<int>({3, 3}), data);
  std::shared_ptr<Tensor> cov = t.covariance();
  std::vector<float> expectedCovariance({1, 9, 72, 9, 84, 672, 72, 672, 5376});
  REQUIRE(cov->read() == expectedCovariance);
  std::shared_ptr<Tensor> flatcov = t.covariance(true);
  std::vector<float> expectedFlatCovariance({1, 9, 72, 84, 672, 5376});
  REQUIRE(flatcov->read() == expectedFlatCovariance);
}
