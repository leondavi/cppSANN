/*
 * propagation.cpp
 *
 *  Created on: Dec 7, 2019
 *      Author: david
 */

#include "include/propagation.h"
#include <iostream>
#define DEBUG_FLAG 0
#define DEBUG_FLAG_BP 0

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
		if(DEBUG_FLAG) {
		std::cout<<"Layer type: "<<current_layer->get_layer_type()<<std::endl;
		std::cout<<"Layer: "<<i++<<std::endl;}

		next_layer = current_layer->get_next_layer_ptr().lock();
		if(next_layer)
		{
			std::shared_ptr<ANN::Weights> output_weights_ptr = current_layer->get_output_weights_ptr();

			if(DEBUG_FLAG)
			{
			std::cout<<"Weights: \n"<<*output_weights_ptr->get_weights_mat_ptr()<<std::endl;
			std::cout<<"Neurons: \n"<<*current_layer->get_neurons_ptr()<<std::endl;
			std::cout<<"Bias: \n"<<*output_weights_ptr->get_bias_ptr()<<std::endl;
			}


			VectorXd Wx_val = output_weights_ptr->dot(*current_layer->get_neurons_ptr());

			if(DEBUG_FLAG)
			{
			std::cout<<"Dot result Wx: \n"<<Wx_val<<std::endl;
			}

			Wx_val = Wx_val + (*output_weights_ptr->get_bias_ptr())*VectorXd::Ones(Wx_val.rows());

			if(DEBUG_FLAG)
			{
			std::cout<<"Result with bias Wx+b: \n"<<Wx_val<<std::endl;
			}

			VectorXd Wx_val_act = Wx_val.unaryExpr(next_layer->get_activation_func_ptr()->get_func());

			if(DEBUG_FLAG)
			{
			std::cout<<"f(Wx+b): \n"<<Wx_val_act<<std::endl;
			}

			next_layer->set_new_val_to_neurons(Wx_val_act);

			if(DEBUG_FLAG)
			{
			std::cout<<"===================================\n"<<std::endl;
			}

		}
		else
		{
			throw std::runtime_error("Forward propagation couldn't lock next layer!");
		}
		current_layer = next_layer;
	}

	return true;
}

/************************************/
/* 		BackwardPropagation 		*/
/************************************/

bool BackwardPropagation::execute(VectorXd Y)
{
	std::cout<<"--------------------------\nBackward Prop\n--------------------------"<<std::endl;

	std::shared_ptr<Layer> current_layer;
	std::shared_ptr<Layer> previous_layer;

	current_layer = this->output_layer_ptr_;

	error_ = this->loss_func_->func(*(current_layer->get_neurons_ptr()),Y).sum();

	VectorXd dEtot_dout = this->loss_func_->derivative(*(current_layer->get_neurons_ptr()),Y);

	int l = 0;

	while((current_layer->get_layer_type() != INPUT_LAYER) && current_layer->get_has_previous())
	{
		if(DEBUG_FLAG_BP)
		{
			std::cout<<"Layer: "<<l<<"\n==========================\n";
		}
		VectorXd dout_dnet = current_layer->get_neurons_ptr()->unaryExpr(current_layer->get_activation_func_ptr()->get_Dfunc());

		previous_layer = current_layer->get_previous_layer_ptr().lock();

		if(DEBUG_FLAG_BP)
				{
		std::cout<<"current neurons:\n"<<*(current_layer->get_neurons_ptr())<<std::endl;
		std::cout<<"dEtot_dout:\n"<<dEtot_dout<<std::endl;
		std::cout<<"dout_dnet:\n"<<dout_dnet<<std::endl;
				}


		if(previous_layer)
		{
			if(DEBUG_FLAG_BP)
					{
			std::cout<<"Previous Neurons: \n"<<*(previous_layer->get_neurons_ptr())<<std::endl;
					}
			//initialize the size of gradient (grad value per each weight)
			MatrixXd weights_grad(current_layer->get_input_weights_ptr()->get_weights_mat_ptr()->rows(),
								  current_layer->get_input_weights_ptr()->get_weights_mat_ptr()->cols());


			double bias_diff;

			VectorXd dcurr_dprev(dout_dnet.size());

			for (uint32_t row = 0; row < weights_grad.rows(); row++)
			{
				double etot_dout_sc = dEtot_dout(row);
				double dout_dnet_sc = dout_dnet(row);
				dcurr_dprev(row) = etot_dout_sc*dout_dnet_sc;
				weights_grad.row(row) = *(previous_layer->get_neurons_ptr())*etot_dout_sc*dout_dnet_sc;
				bias_diff += etot_dout_sc*dout_dnet_sc;
			}

			if(DEBUG_FLAG_BP)
					{
			std::cout<<"weights diff: \n"<<weights_grad<<std::endl;
			std::cout<<"weights_mat: \n"<<*(current_layer->get_input_weights_ptr()->get_weights_mat_ptr())<<std::endl;
					}

			current_layer->get_optimizer()->optimize(*(current_layer->get_input_weights_ptr()->get_weights_mat_ptr()),weights_grad,
													 *(current_layer->get_input_weights_ptr()->get_bias_ptr()),bias_diff,lr_);


			if(DEBUG_FLAG_BP)
					{
			std::cout<<"weights_mat after optimize: \n"<<*(current_layer->get_input_weights_ptr()->get_weights_mat_ptr())<<std::endl;
					}

			current_layer = previous_layer;

			if(DEBUG_FLAG_BP)
					{
			std::cout<<"dcurr_dprev\n"<<dcurr_dprev<<std::endl;
					}
			dEtot_dout = current_layer->get_output_weights_ptr()->get_weights_mat_ptr()->transpose()*dcurr_dprev;
			if(DEBUG_FLAG_BP)
					{
			std::cout<<"dEtot_dout\n"<<dEtot_dout<<std::endl;
					}

			l++;
		}
	}
	//end while
//
//	while((current_layer->get_layer_type() != INPUT_LAYER) && current_layer->get_has_previous())
//	{
//		previous_layer = current_layer->get_previous_layer_ptr().lock();
//		if(previous_layer)
//		{
//
//
//			current_layer = previous_layer;
//		}
//	}

	return true;
}


}

}


