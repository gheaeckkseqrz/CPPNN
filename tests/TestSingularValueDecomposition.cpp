#include "SingularValueDecomposition.h"
#include "TorchLoader.h"

TEST_CASE( "SVD" )
{
  std::shared_ptr<NN::Tensor> covariance = std::make_shared<NN::Tensor>(std::vector<int>({3, 3}), std::vector<float>({
	0.100042433, 0.0741936192, 0.0519854389,
	  0.0741936192, 0.0730947107, 0.0691589117,
	  0.0519854389, 0.0691589117, 0.0864165947}));
  NN::SingularValueDecomposition svd(covariance);

  std::vector<float> u = svd.getU()->read();
  std::vector<float> groundTruthU({0.6079,  0.6917,  0.3898, 0.5758, 0.0461, 0.8163, 0.5467, 0.7207,  0.4263});
  for (int i(0) ; i < 9 ; ++i)
    REQUIRE( abs(u[i]) == Approx(groundTruthU[i]).margin(0.001) ); // Some vector are flipped compared to Torch generated ground truth
  std::vector<float> v = svd.getValue()->read();
  std::vector<float> groundTruthV({ 0.2171, 0.0409, 0.0015});
  for (int i(0) ; i < 3 ; ++i)
    REQUIRE( abs(v[i]) == Approx(groundTruthV[i]).margin(0.001) ); // Some vector are flipped compared to Torch generated ground truth
}
