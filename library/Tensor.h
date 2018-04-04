#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <memory>
#include <vector>
#include "cl.hpp"

#include "Storage.h"

namespace NN
{
  class Tensor
  {
  public:
    Tensor(cl_mem_flags flags =  CL_MEM_READ_WRITE);
    Tensor(std::vector<int> const &sizes, cl_mem_flags flags = CL_MEM_READ_WRITE);
    Tensor(std::vector<float> const &data, cl_mem_flags flags = CL_MEM_READ_WRITE);
    Tensor(std::vector<int> const &sizes, std::vector<float> const &data, cl_mem_flags flags = CL_MEM_READ_WRITE);
    Tensor(Tensor const &o);
    virtual ~Tensor();

    int getSize(int index) const;
    std::vector<int> const &getSizes() const;
    void setSizes(std::vector<int> const &sizes);
    void setOffset(size_t offset);
    void flatten();
    std::vector<float> read() const;
    cl::Buffer getBuffer() const;
    cl::Buffer getSizesBuffer() const;
    size_t getOffset() const;
    size_t getNbElements() const;
    bool dataEquals(Tensor const &o, float tolerance = 0.00001) const;

    Tensor &add(float value);
    Tensor &sub(float value);
    Tensor &mul(float value);
    Tensor &div(float value);
    Tensor &add(Tensor const &o);
    Tensor &sub(Tensor const &o);
    Tensor &mul(Tensor const &o);
    Tensor &div(Tensor const &o);
    Tensor &fill(float value);
    Tensor &copy(Tensor const &o);
    Tensor &copy(std::vector<float> const &o);

    Tensor means() const;
    float min() const;
    float max() const;
    Tensor transpose() const;
    std::shared_ptr<Tensor> covariance(); // const;
    std::shared_ptr<Tensor> matrixMultiply(Tensor &o, std::shared_ptr<Tensor> output = nullptr);

    std::string print(bool data = false) const;

    Tensor operator[](int index);
    Tensor operator[](std::pair<int, int> range);

  protected:
    size_t _offset;

  public:
    std::shared_ptr<Storage<float>> _storage;
    std::unique_ptr<Storage<int>> _sizesStorage;

  private:

    std::vector<int> _sizes;


  private:
    // Private copy constructor to avoid leaks
    /* Tensor(Tensor &o) { std::cerr << "Something called the copy constructor" << std::endl; } */
    /* Tensor(Tensor const &o) { std::cerr << "Something called the copy constructor" << std::endl; } */
  };

  std::ostream &operator<<(std::ostream &s, Tensor const &t);
}

#endif
