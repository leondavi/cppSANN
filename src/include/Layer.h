/*
 * Layer.h
 *
 *  Created on: Dec 6, 2019
 *      Author: David Leon
 */

#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <eigen3/Eigen/Dense>
#include <stdlib.h>
#include <memory>

#include "Weights.h"


using namespace Eigen;

namespace ANN
{

enum {NORMAL_LAYER,INPUT_LAYER,OUTPUT_LAYER};
/**
 * This class represents a single layer
 */
class Layer : std::enable_shared_from_this<Layer>
{
private:
	uint32_t layer_size_;
	VectorXd neurons_; //column vector
	int layer_type;

	std::weak_ptr<Layer> previous_layer_ptr_;
	std::weak_ptr<Layer> next_layer_ptr_;
	std::shared_ptr<Weights> input_weights_ptr_;
	std::shared_ptr<Weights> output_weights_ptr_;

	std::shared_ptr<Layer> getptr() { return std::shared_ptr<Layer>(this); }


public:

	Layer(uint32_t layer_size,std::weak_ptr<Layer> previous_layer_ptr,std::weak_ptr<Layer> next_layer_ptr) :
		layer_size_(layer_size),neurons_(layer_size),layer_type(NORMAL_LAYER),
		previous_layer_ptr_(previous_layer_ptr),next_layer_ptr_(next_layer_ptr)
	{

	}
	virtual ~Layer() {}

	//---- getters ----//
	virtual inline std::weak_ptr<Layer> get_previous_layer_ptr() { return this->previous_layer_ptr_; }
	virtual inline std::weak_ptr<Layer> get_next_layer_ptr() { return this->next_layer_ptr_; }
	virtual inline std::shared_ptr<Weights> get_input_weights_ptr() { return this->input_weights_ptr_; }
	virtual inline std::shared_ptr<Weights> get_output_weights_ptr() { return this->output_weights_ptr_; }

	inline uint32_t get_layer_size() { return this->layer_size_; }
	virtual inline int get_layer_type() { return this->layer_type; }

	bool get_has_next();


	virtual bool connect_next(std::weak_ptr<Layer> next_layer);

	//---- setters ----//

	virtual inline void set_previous_layer_ptr(std::weak_ptr<Layer> previous_layer_ptr) { this->previous_layer_ptr_ = previous_layer_ptr; }
	virtual inline void set_input_weights(std::shared_ptr<Weights> input_weight) { this->input_weights_ptr_ = input_weight; }
	virtual inline void set_output_weights(std::shared_ptr<Weights> output_weight) { this->output_weights_ptr_ = output_weight; }




};//end of Layer


class InputLayer :  public Layer
{
private:

	int layer_type;

public:

	InputLayer(uint32_t layer_size,std::weak_ptr<Layer> next_layer);

	//---- getters ----//

	inline std::weak_ptr<Layer> get_previous_layer_ptr() override { return std::weak_ptr<Layer>(); }//No previous in this layer
	inline std::shared_ptr<Weights> get_input_weights_ptr() override { return std::shared_ptr<Weights>(); }//No input weights to this layer
	inline int get_layer_type() override { return this->layer_type; }


	inline void set_previous_layer_ptr(std::weak_ptr<Layer> previous_layer_ptr) override { Layer::set_previous_layer_ptr(std::weak_ptr<Layer>()); }
	inline void set_input_weights(std::shared_ptr<Weights> input_weight) { throw std::runtime_error("No input weights for InputLayer instance"); }


};//end of InputLayer


class OutputLayer :  public Layer
{
private:

	int layer_type;

public:
	OutputLayer(uint32_t layer_size,std::weak_ptr<Layer> previous_layer);

	//---- getters ----//

	inline std::weak_ptr<Layer> get_next_layer_ptr() override { return std::weak_ptr<Layer>(); }//No next layer after the output layer
	inline std::shared_ptr<Weights> get_output_weights_ptr() override { return std::shared_ptr<Weights>(); }//No output weights to this layer
	inline int get_layer_type() override { return this->layer_type; }

	inline bool connect_next(std::weak_ptr<Layer> next_layer,
								  std::shared_ptr<Weights> output_weights_ptr_ =  std::shared_ptr<Weights>())
	{
		throw std::runtime_error("No connect next for OutpoutLayer");
	}

	inline void set_output_weights(std::shared_ptr<Weights> output_weight) { throw std::runtime_error("No output weights for OutputLayer instance"); }


};//end of InputLayer

}//end of namespace ANN

#endif /* SRC_LAYER_H_ */
