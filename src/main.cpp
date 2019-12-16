

#include "include/definitions.hh"
#include "include/Layer.h"
#include "include/activation_functions.h"

int main(int ac, char** av);
/**
 * Examples how to use
 */

int main(int ac, char** av)
{
	MatrixXd m(4,3);
	m << 1, 2, 3,
	     4, 5, 6,
	     7, 8, 9,
		 10,11,12;
	ANN::Weights new_weight(m);
	VectorXd sample_vec(3); sample_vec << 1,2,3;
	std::cout<<"Before: "<<*new_weight.get_weights_mat()<<std::endl;
	VectorXd dprod_vec = new_weight.dot(sample_vec)/100;

	std::cout<<"dprod vec: \n"<<dprod_vec<<std::endl;

	dprod_vec = dprod_vec.unaryExpr(&Activations::Sigmoid);


//	std::cout<<"After: "<<*new_weight.get_weights_mat()<<std::endl;
//	std::cout<<"sample_vec: \n"<<sample_vec<<std::endl;
	std::cout<<"dprod vec: \n"<<dprod_vec<<std::endl;


	return 0;

}
