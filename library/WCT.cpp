#include "SingularValueDecomposition.h"
#include "WCT.h"

namespace NN
{
  WCT::WCT()
  {
  }

  std::shared_ptr<Tensor> WCT::enforceCovariance(std::shared_ptr<Tensor> data, std::shared_ptr<Tensor> targetCovariance, std::shared_ptr<Tensor> targetMean)
  {
    std::shared_ptr<Tensor> whitenedData = whitenData(data);
    SingularValueDecomposition svd(targetCovariance);
    std::shared_ptr<Tensor> s_d1 = svd.getValue();
    s_d1->sqrt();
    std::shared_ptr<Tensor> cd1 = svd.getU()->matrixMultiply(*(s_d1->diagonalise()));
    Tensor vectorTranspose = svd.getU()->transpose();
    std::shared_ptr<Tensor> cd2 = cd1->matrixMultiply(vectorTranspose);
    std::shared_ptr<Tensor> cd3 = cd2->matrixMultiply(*whitenedData);
    cd3->add(*targetMean);
    return cd3;
  }

  std::shared_ptr<Tensor> WCT::whitenData(std::shared_ptr<Tensor> data)
  {
    std::shared_ptr<Tensor> covariance = data->covariance(false, true);
    data->sub(data->means());
    SingularValueDecomposition svd(covariance);
    std::shared_ptr<Tensor> c_d = svd.getValue();
    c_d->sqrt();
    c_d->pow(-1);
    std::shared_ptr<Tensor> wc1 = svd.getU()->matrixMultiply(*(c_d->diagonalise()));
    Tensor vectorTranspose = svd.getU()->transpose();
    std::shared_ptr<Tensor> wc2 = wc1->matrixMultiply(vectorTranspose);
    std::shared_ptr<Tensor> wc3 = wc2->matrixMultiply(*data);
    return wc3;
  }
}
