/*
 * propagation.cpp
 *
 *  Created on: Dec 7, 2019
 *      Author: david
 */

#include "include/propagation.h"

namespace ANN
{

namespace Propagation
{


/************************************/
/* 		ForwardPropagation 			*/
/************************************/



bool ForwardPropagation::execute()
{
	std::shared_ptr<Layer> current_layer;
	std::shared_ptr<Layer> next_layer;

	current_layer = this->input_layer_ptr_;

	while ((current_layer->get_layer_type() != OUTPUT_LAYER) && current_layer->get_has_next())
	{
		std::shared_ptr<ANN::Weights> output_weights_ptr = current_layer->get_output_weights_ptr();
		next_layer = current_layer->get_next_layer_ptr().lock();
		if(next_layer)
		{
				//TODO make the math
		}
	}

	return true;
}

}

}


