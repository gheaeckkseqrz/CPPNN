#include "SingularValueDecomposition.h"
#include <Eigen/SVD>

namespace NN
{
  SingularValueDecomposition::SingularValueDecomposition(std::shared_ptr<Tensor> matrix)
  {
    std::vector<float> data = matrix->read();
    Eigen::Map<Eigen::MatrixXf> eigenMatrix(data.data(), matrix->getSize(1), matrix->getSize(0));

    Eigen::BDCSVD<Eigen::MatrixXf> svd(eigenMatrix, Eigen::ComputeFullU);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U = svd.matrixU();
    _u = std::make_shared<Tensor>(std::vector<int>({(int)U.rows(), (int)U.cols()}), std::vector<float>(U.data(), U.data() + U.size()));
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V = svd.singularValues();
    _v = std::make_shared<Tensor>(std::vector<int>({(int)V.rows()}), std::vector<float>(V.data(), V.data() + V.size()));
  }

  std::shared_ptr<Tensor> SingularValueDecomposition::getU() const
  {
    return _u;
  }

  std::shared_ptr<Tensor> SingularValueDecomposition::getValue() const
  {
    return _v;
  }

}
