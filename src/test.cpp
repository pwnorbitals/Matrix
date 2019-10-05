#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <tuple>
#include <fstream>
#include <thread>
#include <mutex>
#include <random>
#include <iomanip>

#include "matrix.h"
#include "matrix_test.h"
#include "algos.h"

unsigned tries = 20;
std::vector<unsigned int> sizes {5, 10, 25, 50, 75, 100, 250, 500, 1000, 5000, 10000};
std::vector<long double> precisions {1e-15};
long double min_gen = -1e9;
long double max_gen = 1e9;

using T = long double;

void run_gaussSeidel_bench() {

	std::ofstream file;
	file.open("gauss-seidel_bench.txt", std::ios::out | std::ios::trunc);
	file << std::scientific;

	std::ofstream file2;
	file2.open("gauss-seidel_bench_min.txt", std::ios::out | std::ios::trunc);
	file2 << std::scientific;

	file << "index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val" << "\n";
	file2 << "index, matrix_size, precision, error, time, iterations, min_val, max_val" << "\n";

	for (unsigned s = 0; s < sizes.size(); s++) {
		unsigned int cursize(sizes[s]);
		std::cout << "GS : TESTING SIZE " << cursize << "\n";
		for (unsigned p = 0; p < precisions.size(); p++) {
			T curprec(precisions[p]);
			std::cout << "GS : TESTING PRECISION " << curprec << "\n";
			for (unsigned i = 0; i < tries; i++) {
				auto m = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> A = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> b = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);
				Matrix<T> x0 = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);

				// DOMINANT DIAGONAL
				//for (unsigned i = 0; i < cursize; i++) {
				//	A[i][i] *= max_gen;
				//}


				// START MEASURE TIME
				auto begin(std::chrono::high_resolution_clock::now());

				// CALL FUNCTION
				std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> res0(gaussSeidel(A));
				std::tuple<Matrix<T>, long long, T> result(iter_generale(std::get<1>(res0), std::get<2>(res0), b, x0, curprec, (long long)1e9));

				// END MEASURE TIME
				auto end(std::chrono::high_resolution_clock::now());

				auto duration(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());

				// index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val
				file << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< A.str() << ", "
					<< b.str() << ", "
					<< std::get<0>(result).str() << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";

				file2 << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";
			}
		}
	}

	std::cout << "GS : JOB DONE\n";
	file.close();
}

void run_jacobi_bench() {

	std::ofstream file;
	file.open("jacobi_bench.txt", std::ios::out | std::ios::trunc);
	file << std::scientific;

	std::ofstream file2;
	file2.open("jacobi_bench_min.txt", std::ios::out | std::ios::trunc);
	file2 << std::scientific;

	file << "index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val" << "\n";
	file2 << "index, matrix_size, precision, error, time, iterations, min_val, max_val" << "\n";

	for (unsigned s = 0; s < sizes.size(); s++) {
		unsigned int cursize(sizes[s]);
		std::cout << "JB : TESTING SIZE " << cursize << "\n";
		for (unsigned p = 0; p < precisions.size(); p++) {
			T curprec(precisions[p]);
			std::cout << "JB : TESTING PRECISION " << curprec << "\n";
			for (unsigned i = 0; i < tries; i++) {
				auto m = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> A = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> b = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);
				Matrix<T> x0 = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);

				// DOMINANT DIAGONAL
				//for (unsigned i = 0; i < cursize; i++) {
				//	A[i][i] *= max_gen;
				//}

				// START MEASURE TIME
				auto begin(std::chrono::high_resolution_clock::now());

				// CALL FUNCTION
				std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> res0(jacobi(A));
				std::tuple<Matrix<T>, long long, T> result(iter_generale(std::get<1>(res0), std::get<2>(res0), b, x0, curprec, (long long)1e9));

				// END MEASURE TIME
				auto end(std::chrono::high_resolution_clock::now());

				auto duration(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());

				// index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val
				file << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< A.str() << ", "
					<< b.str() << ", "
					<< std::get<0>(result).str() << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";

				file2 << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";
			}
		}
	}

	std::cout << "JB : JOB DONE\n";
	file.close();
}

void run_richardson_bench() {

	std::ofstream file;
	file.open("richardson_bench.txt", std::ios::out | std::ios::trunc);
	file << std::scientific;

	std::ofstream file2;
	file2.open("richardson_bench_min.txt", std::ios::out | std::ios::trunc);
	file2 << std::scientific;

	file << "index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val" << "\n";
	file2 << "index, matrix_size, precision, error, time, iterations, min_val, max_val" << "\n";

	for (unsigned s = 0; s < sizes.size(); s++) {
		unsigned int cursize(sizes[s]);
		std::cout << "RD : TESTING SIZE " << cursize << "\n";
		for (unsigned p = 0; p < precisions.size(); p++) {
			T curprec(precisions[p]);
			std::cout << "RD : TESTING PRECISION " << curprec << "\n";
			for (unsigned i = 0; i < tries; i++) {
				auto m = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> A = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> b = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);
				Matrix<T> x0 = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);

				// DOMINANT DIAGONAL
				for (unsigned i = 0; i < cursize; i++) {
					A[i][i] *= max_gen;
				}

				// START MEASURE TIME
				auto begin(std::chrono::high_resolution_clock::now());

				// CALL FUNCTION
				throw std::logic_error("RICHARDSON PAS DEFINI FDP");
				std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> res0(jacobi(A));
				std::tuple<Matrix<T>, long long, T> result(iter_generale(std::get<1>(res0), std::get<2>(res0), b, x0, curprec, (long long)1e9));

				// END MEASURE TIME
				auto end(std::chrono::high_resolution_clock::now());

				auto duration(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());

				// index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val
				file << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< A.str() << ", "
					<< b.str() << ", "
					<< std::get<0>(result).str() << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";

				file2 << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";
			}
		}
	}

	std::cout << "RD : JOB DONE\n";
	file.close();
}

void run_sor_bench() {

	std::ofstream file;
	file.open("sor_bench.txt", std::ios::out | std::ios::trunc);
	file << std::scientific;

	std::ofstream file2;
	file2.open("sor_bench_min.txt", std::ios::out | std::ios::trunc);
	file2 << std::scientific;

	file << "index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val" << "\n";
	file2 << "index, matrix_size, precision, error, time, iterations, min_val, max_val" << "\n";

	for (unsigned s = 0; s < sizes.size(); s++) {
		unsigned int cursize(sizes[s]);
		std::cout << "SR : TESTING SIZE " << cursize << "\n";
		for (unsigned p = 0; p < precisions.size(); p++) {
			T curprec(precisions[p]);
			std::cout << "SR : TESTING PRECISION " << curprec << "\n";
			for (unsigned i = 0; i < tries; i++) {
				auto m = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> A = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> b = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);
				Matrix<T> x0 = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);

				// DOMINANT DIAGONAL
				//for (unsigned i = 0; i < cursize; i++) {
				//	A[i][i] *= max_gen;
				//}

				// SETTINGS
				std::random_device dev;
				std::mt19937 rng(dev());
				std::uniform_real_distribution<T> distribution(0, 2);
				T omega = distribution(rng);

				// START MEASURE TIME
				auto begin(std::chrono::high_resolution_clock::now());

				// CALL FUNCTION
				std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> res0(sor(A, omega));
				std::tuple<Matrix<T>, long long, T> result(iter_generale(std::get<1>(res0), std::get<2>(res0), b, x0, curprec, (long long)1e9));

				// END MEASURE TIME
				auto end(std::chrono::high_resolution_clock::now());

				auto duration(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());

				// index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val
				file << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< A.str() << ", "
					<< b.str() << ", "
					<< std::get<0>(result).str() << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";

				file2 << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";
			}
		}
	}

	std::cout << "SR : JOB DONE\n";
	file.close();
}

void run_gmres_bench() {
	/*
	std::ofstream file;
	file.open("gmres_bench.txt", std::ios::out | std::ios::trunc);
	file << std::scientific;
	*/

	std::ofstream file2;
	file2.open("gmres_bench_min.txt", std::ios::out | std::ios::trunc);
	file2 << std::scientific;

	//file << "index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val" << "\n";
	file2 << "index, matrix_size, precision, error, time, iterations, min_val, max_val" << "\n";

	for (unsigned s = 0; s < sizes.size(); s++) {
		unsigned int cursize(sizes[s]);
		std::cout << "GR : TESTING SIZE " << cursize << "\n";
		for (unsigned p = 0; p < precisions.size(); p++) {
			T curprec(precisions[p]);
			std::cout << "GR : TESTING PRECISION " << curprec << "\n";
			for (unsigned i = 0; i < tries; i++) {
				auto m = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> A = Matrix<T>::gen_random(cursize, min_gen, max_gen);
				Matrix<T> b = Matrix<T>::gen_random(cursize, 1, min_gen, max_gen);

				// DOMINANT DIAGONAL
				//for (unsigned i = 0; i < cursize; i++) {
				//	A[i][i] *= max_gen;
				//}

				// START MEASURE TIME
				auto begin(std::chrono::high_resolution_clock::now());

				// CALL FUNCTION
				std::tuple<Matrix<T>, long long, T> result(gmres(A, b, curprec));

				// END MEASURE TIME
				auto end(std::chrono::high_resolution_clock::now());

				auto duration(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());

				// index, matrix_size, precision, error, time, A, b, x, iterations, min_val, max_val
				std::cout << i << std::endl;
				/*
				file << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< A.str() << ", "
					<< b.str() << ", "
					<< std::get<0>(result).str() << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";
					*/

				file2 << i << ", "
					<< cursize << ", "
					<< curprec << ", "
					<< std::get<2>(result) << ", "
					<< duration << ", "
					<< std::get<1>(result) << ", "
					<< min_gen << ", "
					<< max_gen << ", "
					<< "\n";
			}
		}
	}

	std::cout << "GR : JOB DONE\n";
	//file.close();
	file2.close();
}



int main() {

	Matrix<T>::run_tests();
	test_algos<T>();
	
	
	std::vector<std::thread> vect;

	//vect.push_back(std::thread(run_gaussSeidel_bench));
	//vect.push_back(std::thread(run_jacobi_bench));
	//run_richardson_bench();
	//vect.push_back(std::thread(run_sor_bench));
	vect.push_back(std::thread(run_gmres_bench));

	for (size_t i = 0; i < vect.size(); i++) {
		vect[i].join();
	}
	
	
	system("pause");
}
