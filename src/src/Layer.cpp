/*
 * Layer.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: david
 */

#include "../include/Layer.h"


//************* Layer ************//











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
