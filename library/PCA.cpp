#include <cassert>
#include <clBLAS.h>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include "PCA.h"

namespace NN
{
  GPUOp::GPUOp(std::shared_ptr<Tensor> t)
    :_t(t)
  {
    assert(t->getSize(0) == t->getSize(1));
    _in = std::make_shared<Tensor>(std::vector<int>({t->getSize(0)}));
    _out = std::make_shared<Tensor>(std::vector<int>({t->getSize(0)}));
  }

  int GPUOp::rows()
  {
    return _t->getSize(0);
  }

  int GPUOp::cols()
  {
    return _t->getSize(1);
  }

  // Perform a matrix vector operation.
  // x_in is the vector
  // _t is the matrix
  // y_out is the result buffer
  void GPUOp::perform_op(float *x_in, float *y_out)
  {
    _in->copy(std::vector<float>(x_in, x_in + _t->getSize(0)));
    cl_command_queue queue = OpenCL::getInstance()->getQueue()();
    clblasSgemv(clblasRowMajor, clblasNoTrans,
		_t->getSize(0), _t->getSize(1),
		1.0f, _t->getBuffer()(), _t->getOffset(), _t->getSize(1),
		_in->getBuffer()(), _in->getOffset(), 1,
		1.0f, _out->getBuffer()(), _out->getOffset(), 1,
		1, &queue, 0, NULL, NULL);
    std::vector<float> res = _out->read();
    std::memcpy(y_out, res.data(), res.size() * sizeof(float));
  }

  PCA::PCA()
  {
  }

  std::shared_ptr<Tensor> PCA::reduceDataToNbOfDims(std::shared_ptr<Tensor> data, int dims)
  {
    std::vector<int> dataSizesBackup = data->getSizes();
    data->flatten();
    std::shared_ptr<Tensor> covarianceMatrix = data->covariance();
    std::shared_ptr<Tensor> res = std::make_shared<Tensor>(std::vector<int>(dims, data->getSize(1)));

    // std::vector<float> v = data->read();
    // Eigen::MatrixXf A = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(v.data(), data->getSize(0), data->getSize(1));
    // Spectra::DenseSymMatProd<float> op(A);
    GPUOp op(covarianceMatrix);
    Spectra::SymEigsSolver<float, Spectra::LARGEST_ALGE, GPUOp> eigs(&op, dims, std::max(dims * 2, 20));
    eigs.init();
    eigs.compute();
    if(eigs.info() == Spectra::SUCCESSFUL)
      {
    	Eigen::MatrixXf evector = eigs.eigenvectors();
    	float *eigenVectorsData = evector.data();
    	std::vector<float> eigenDataVector(eigenVectorsData, eigenVectorsData + (data->getSize(0) * dims));

    	std::shared_ptr<Tensor> eigenVectorTensor = std::make_shared<Tensor>(std::vector<int>({dims, data->getSize(0)}), eigenDataVector);
	res = eigenVectorTensor->matrixMultiply(*data);
      }
    else
      throw std::runtime_error("Eigendecomposition failled");
    data->setSizes(dataSizesBackup);
    std::vector<int> resSizes = data->getSizes();
    resSizes[0] = dims;
    res->setSizes(resSizes);
    return res;
  }
}
