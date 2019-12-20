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
		next_layer = current_layer->get_next_layer_ptr().lock();
		if(next_layer)
		{
			std::shared_ptr<ANN::Weights> output_weights_ptr = current_layer->get_output_weights_ptr();

			VectorXd Wx_val = output_weights_ptr->dot(*current_layer->get_neurons_ptr());
			Wx_val = Wx_val + output_weights_ptr->get_bias()*VectorXd::Ones(Wx_val.rows());
			VectorXd Wx_val_act = Wx_val.unaryExpr(next_layer->get_activation_func());
			next_layer->set_new_val_to_neurons(Wx_val_act);
		}
	}

	return true;
}

}

}


