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



namespace Propagation
{

class ForwardPropagation
{

private:

	std::shared_ptr<InputLayer> input_layer_ptr_;

	bool validity_check();

public:

	ForwardPropagation(std::shared_ptr<InputLayer> input_layer_ptr) : input_layer_ptr_(input_layer_ptr)
	{

	}

	bool execute();


};


class BackwardPropagation
{

private:


public:


};

}//end of namespace

#endif /* SRC_INCLUDE_PROPAGATION_H_ */
