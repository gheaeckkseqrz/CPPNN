#include <iostream>
#include "Node.h"
#include "../MatToTensor.h"
#include "Placeholder.h"
#include "TensorflowParser.h"

#define NETWORK_FILE "/Users/wilmot_p/Desktop/optimized_model.meta"

int main(int ac, char **av)
{
  std::cout << "Neural Upres" << std::endl;
  std::cout << "Semantic texture segmentation" << std::endl;
  if (ac < 2)
    {
      std::cerr << "Usage : " << av[0] << " INPUT_IMAGE" << std::endl;
      return 1;
    }

  std::shared_ptr<NN::Tensor> inputImage = tensorFromImage(av[1]);
  std::cout << "Input : " << *inputImage << std::endl;
  NN::ComputeGraph graph;
  try
    {
      graph = NN::TensorflowParser::getInstance()->loadFromFile(NETWORK_FILE);
    }
  catch (const std::runtime_error& e)
    {
      std::cerr << "Exception while loading network : [" << e.what() << "]" << std::endl;
      return -1;
    }

  inputImage->div(255);
  inputImage->mul(2);
  inputImage->sub(1.0f);
  std::shared_ptr<NN::Placeholder> inputNode = std::dynamic_pointer_cast<NN::Placeholder>(graph.getNode("import/input_image"));
  inputNode->setContent(inputImage);
  std::shared_ptr<NN::Node> outputNode = graph.getNode("import/SRGAN_g_2242243/out/Tanh");
  std::shared_ptr<NN::Tensor> outputImage = outputNode->evaluate(graph);
  outputImage->add(1.0f);
  outputImage->mul(2.0f);
  outputImage->mul(255);
  saveTensorAsImage(outputImage, "out.png");
  return 0;
}
