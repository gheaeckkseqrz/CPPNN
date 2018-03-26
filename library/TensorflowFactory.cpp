#include "TensorflowFactory.h"

#include "NodeAdd.h"
#include "NodeMul.h"
#include "PixelShuffle.h"
#include "Placeholder.h"
#include "Relu.h"
#include "Tanh.h"
#include "TensorflowConvolution.h"
#include "TensorflowParser.h"
#include "TensorflowTensor.h"

namespace NN
{
  TensorflowFactory *TensorflowFactory::_instance = nullptr;

  TensorflowFactory::TensorflowFactory()
  {
  }

  TensorflowFactory *TensorflowFactory::getInstance()
  {
    if (_instance == nullptr)
      _instance = new TensorflowFactory();
    return _instance;
  }

  std::shared_ptr<Node> TensorflowFactory::createNode(std::string const &s)
  {
    std::string name = TensorflowParser::getInstance()->getAttribute("name", s);
    std::string op = TensorflowParser::getInstance()->getAttribute("op", s);
    std::list<std::pair<std::string, std::string>> attributes = TensorflowParser::getInstance()->getAttributeList(s);
    std::list<std::string> inputs;
    for (std::pair<std::string, std::string> t : attributes)
      if (t.first == "input")
	if (t.second[0] != '^')
	  inputs.push_back(TensorflowParser::getInstance()->stripQuotes(t.second));

    if (op == "Add" || op == "BiasAdd")
      return std::make_shared<NodeAdd>(inputs);
    else if (op == "Conv2D")
      return std::make_shared<TensorflowConvolution>(inputs);
    else if (op == "Mul")
      return std::make_shared<NodeMul>(inputs);
    else if (op == "Placeholder")
      return std::make_shared<Placeholder>();
    else if (op == "Relu")
      return std::make_shared<Node>(inputs, std::make_shared<Relu>());
    else if (op == "Tanh")
      return std::make_shared<Node>(inputs, std::make_shared<Tanh>());
    else if (op == "Reshape")
      {
    	// Dirty hack where we discard most of the nodes composing A pixel Shuffle operation
    	// We end up processing only the reshape node, and pretend Split, ConcatV2, Shape, StridedSlice and Pack don't exist
    	inputs.pop_front();
    	return std::make_shared<Node>(inputs, std::make_shared<PixelShuffle>());
      }
    else if (op == "Const")
      {
    	std::shared_ptr<Tensor> content(nullptr);
    	std::list<std::pair<std::string, std::string>> elements = TensorflowParser::getInstance()->getElementsList(s);
    	for (std::pair<std::string, std::string> e : elements)
    	  {
    	    std::string elementKey = TensorflowParser::getInstance()->getAttribute("key", e.second);
    	    if (elementKey == "value")
    	      content = std::make_shared<TensorflowTensor>(e.second);
    	  }
    	return std::make_shared<Placeholder>(content);
      }
    else
      std::cerr << "Unimplemented operation [" << op << "]" << std::endl;
    return std::make_shared<Node>(inputs);
  }

}
