/*
 * propagation.h
 *
 *  Created on: Dec 7, 2019
 *      Author: david
 */

#ifndef SRC_INCLUDE_PROPAGATION_H_
#define SRC_INCLUDE_PROPAGATION_H_

#include <memory>
#include "loss_functions.h"
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
	LossFunctionPtr loss_func_;

public:

	BackwardPropagation(std::shared_ptr<OutputLayer> output_layer_ptr,
						double learning_rate,
						LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
		output_layer_ptr_(output_layer_ptr),
		lr_(learning_rate),
		loss_func_(loss_func)
	{

	}

	bool execute(VectorXd Y);
};

}//end of namespace Propagation

}//end of namespace ANN

#endif /* SRC_INCLUDE_PROPAGATION_H_ */
