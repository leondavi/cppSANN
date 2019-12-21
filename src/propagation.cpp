/*
 * propagation.cpp
 *
 *  Created on: Dec 7, 2019
 *      Author: david
 */

#include "include/propagation.h"
#include <iostream>

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

	int i=1; //TODO remove
	while ((current_layer->get_layer_type() != OUTPUT_LAYER) && current_layer->get_has_next())
	{
		std::cout<<"Layer type: "<<current_layer->get_layer_type()<<std::endl;
		std::cout<<"Layer: "<<i++<<std::endl;
		next_layer = current_layer->get_next_layer_ptr().lock();
		if(next_layer)
		{
			std::shared_ptr<ANN::Weights> output_weights_ptr = current_layer->get_output_weights_ptr();

			std::cout<<"Weights: \n"<<*output_weights_ptr->get_weights_mat()<<std::endl;
			std::cout<<"Neurons: \n"<<*current_layer->get_neurons_ptr()<<std::endl;
			std::cout<<"Bias: \n"<<output_weights_ptr->get_bias()<<std::endl;


			VectorXd Wx_val = output_weights_ptr->dot(*current_layer->get_neurons_ptr());

			std::cout<<"Dot result Wx: \n"<<Wx_val<<std::endl;

			Wx_val = Wx_val + output_weights_ptr->get_bias()*VectorXd::Ones(Wx_val.rows());

			std::cout<<"Result with bias Wx+b: \n"<<Wx_val<<std::endl;

			VectorXd Wx_val_act = Wx_val.unaryExpr(next_layer->get_activation_func());

			std::cout<<"f(Wx+b): \n"<<Wx_val_act<<std::endl;
			next_layer->set_new_val_to_neurons(Wx_val_act);
			std::cout<<"===================================\n"<<std::endl;

		}
		else
		{
			throw std::runtime_error("Forward propagation couldn't lock next layer!");
		}
		current_layer = next_layer;
	}

	return true;
}

}

}


