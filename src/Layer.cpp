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


bool Layer::connect_next(std::weak_ptr<Layer> next_layer,std::weak_ptr<Weights> output_weights_ptr_ )
{

	if (next_layer.expired()) { set_disconnected(); return false; } //check next layer exists
	std::shared_ptr<Layer> next_layer_inst = next_layer.lock();

	this->next_layer_ptr_ = next_layer; //connect to the next layer
	next_layer_inst->set_previous_layer_ptr(std::weak_ptr<Layer>(getptr()));

	// rows ---> Next layer
	// cols ---> Current Layer
	std::shared_ptr<Weights> weights_ptr = std::make_shared<Weights>(next_layer_inst->get_layer_size(),this->get_layer_size());




	return false;
}








//********* InputLayer *************//

InputLayer::InputLayer(uint32_t layer_size,std::weak_ptr<Layer> next_layer) :
						Layer(layer_size,std::weak_ptr<Layer>(),next_layer)
{

}




//********* OutputLayer ***********//
OutputLayer::OutputLayer(uint32_t layer_size,std::weak_ptr<Layer> previous_layer) :
						Layer(layer_size,std::weak_ptr<Layer>(),previous_layer)
{

}

}
