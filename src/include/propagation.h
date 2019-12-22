/*
 * propagation.h
 *
 *  Created on: Dec 7, 2019
 *      Author: david
 */

#ifndef SRC_INCLUDE_PROPAGATION_H_
#define SRC_INCLUDE_PROPAGATION_H_

#include <memory>
#include "Layer.h"

namespace ANN
{

namespace Propagation
{

class ForwardPropagation
{

private:

	std::shared_ptr<InputLayer> input_layer_ptr_;

public:

	ForwardPropagation(std::shared_ptr<InputLayer> input_layer_ptr) : input_layer_ptr_(input_layer_ptr)
	{

	}

	bool execute();


};


class BackwardPropagation
{

private:

	std::shared_ptr<OutputLayer> output_layer_ptr_;
	double lr_;//learning rate

public:

	BackwardPropagation(std::shared_ptr<OutputLayer> output_layer_ptr, double learning_rate) :
		output_layer_ptr_(output_layer_ptr),
		lr_(learning_rate)
	{

	}

	bool execute();
};

}//end of namespace Propagation

}//end of namespace ANN

#endif /* SRC_INCLUDE_PROPAGATION_H_ */
