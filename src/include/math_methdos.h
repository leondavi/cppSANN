#pragma once

#include <eigen3/Eigen/Eigen>
#include <random>

inline Eigen::MatrixXd randn(uint32_t rows, uint32_t cols,double mu = 0.,double sig = 1.)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mu,sig);
	Eigen::MatrixXd newMat(rows,cols);

	for (int row = 0; row < newMat.rows(); row++)
	{
		for (int col=0; col < newMat.cols(); col++)
		{
			newMat(row,col) = distribution(generator);
		}
	}
	return newMat;
}

inline Eigen::VectorXd randn(uint32_t size,double mu=0.,double var=1.)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mu,var);
	Eigen::VectorXd newVec(size);
	for (uint32_t i = 0; i < size; i++)
	{
		newVec(i) = distribution(generator);
	}

	return newVec;
}

/**
 * function changes the value of cell
 */
inline void set_vals_by_func(Eigen::MatrixXd &mat, std::function<double()> &func)
{
	for (int row = 0; row < mat.rows(); row++)
	{
		for (int col=0; col < mat.cols(); col++)
		{
			mat(row,col) = func();
		}
	}
}

/**
 * Function receives current state of cell and change it accordingly
 */
inline void change_vals_by_func(Eigen::MatrixXd &mat,std::function<double(double)> func)
{
	for (int row = 0; row < mat.rows(); row++)
	{
		for (int col=0; col < mat.cols(); col++)
		{
			mat(row,col) = func(mat(row,col));
		}
	}
}


