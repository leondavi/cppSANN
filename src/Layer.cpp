/*
 * Layer.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: david
 */

#include "include/Layer.h"

namespace ANN
{
//************* Layer ************//


bool Layer::connect_next(std::weak_ptr<Layer> next_layer)
{

	std::shared_ptr<Layer> next_layer_inst = next_layer.lock();

	if(next_layer_inst)
	{

		this->next_layer_ptr_ = next_layer; //connect to the next layer
		next_layer_inst->set_previous_layer_ptr(std::weak_ptr<Layer>(getptr()));

		// Creating weights instance
		// rows ---> Next layer
		// cols ---> Current Layer
		// Wx+b where W is NxP and x is Px1 (N-Next layer dimension,P-Previous layer dimension)
		std::shared_ptr<Weights> weights_ptr = std::make_shared<Weights>(next_layer_inst->get_layer_size(),this->get_layer_size());

		next_layer_inst->set_input_weights(weights_ptr);//new weights are the input of next layer
		set_output_weights(weights_ptr);//new weights are the output weights of this layer

		return true;
	}
	return false;
}

//-------- getters ----------//

bool Layer::get_has_next()
{
	std::shared_ptr<Layer> next_layer_inst = this->next_layer_ptr_.lock();
	if(next_layer_inst)
	{
		return true;
	}
	return false;
}

//********* InputLayer *************//

InputLayer::InputLayer(uint32_t layer_size,std::function<double(double)> activation_func,std::weak_ptr<Layer> next_layer) :
						Layer(layer_size,activation_func,std::weak_ptr<Layer>(),next_layer),layer_type(INPUT_LAYER)
{

}




//********* OutputLayer ***********//
OutputLayer::OutputLayer(uint32_t layer_size,std::function<double(double)> activation_func,std::weak_ptr<Layer> previous_layer) :
						Layer(layer_size,activation_func,std::weak_ptr<Layer>(),previous_layer),layer_type(OUTPUT_LAYER)
{

}

}


