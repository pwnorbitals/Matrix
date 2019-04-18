#ifndef MATRIX_DEF_H
#define MATRIX_DEF_H

#include "matrix_test_def.h"

template<class T>
class Matrix {
	friend class matrix_test;

protected:
	std::vector<std::vector<T>> data;

public:
	// CONSTRUCTORS & DESTRUCTORS
	Matrix() : data{ {T{}} } {}; // Implemented
	Matrix(std::vector<std::vector<T>> _data) : data(_data) {}; // Implemented
	// Matrix<T>(std::vector<std::vector<T>>&& _data) : data(_data) {}; // Implemented
	Matrix(size_t const lines, size_t const cols) : data(lines, std::vector<T>(cols, T())) {}; // Implemented
	template<class T2> Matrix(Matrix<T2> const& other) : data(other.data) {}; // Implemented

	// UTILITIES
	size_t cols() const; // Implemented
	size_t lines() const; // Implemented
	std::string str() const; // Implemented
	Matrix<T> line(size_t pos) const; // Implemented
	Matrix<T> col(size_t pos) const; // Implemented
	void setLine(size_t pos, std::vector<T> data); // Implemented
	void setLine(size_t pos, std::vector<T>&& data);
	void setCol(size_t pos, std::vector<T> data); // Implemented
	void setCol(size_t pos, std::vector<T>&& data);
	void setLine(size_t pos, Matrix<T> data); // Implemented
	void setCol(size_t pos, Matrix<T> data); // Implemented
	Matrix<T> truncate(size_t x, size_t y) const; // Implemented
	std::vector<std::vector<T>> const& getData() const; // Implemented


	// OPERATORS
	template<class T2> auto operator+(Matrix<T2> const& other) const; // Implemented
	template<class T2> auto operator-(Matrix<T2> const& other) const; // Implemented
	Matrix<T> operator-() const; // Implemented
	template<class T2> bool operator==(Matrix<T2> const& other) const; // Implemented
	template<class T2> Matrix<T> operator=(Matrix<T2> const& other); // Implemented
	template<class T2> auto operator*(T2 const& other) const; // Implemented
	template<class T2> auto operator/(T2 const& other) const; // Implemented
	auto operator*(T const& other) const; // Implemented
	std::vector<T> & operator[](size_t pos); // Implemented
	T const& get(size_t line, size_t col) const; // Implemented

	// OPERATIONS
	T det() const; // Implemented, tested
	Matrix<T> transp() const; // Implemented, tested
	std::vector<T> diag() const; // Implemented, tested
	Matrix<T> inv_LU() const; // Implemented, tested
	Matrix<T> tri_lo(bool include_diag = false) const; // Implemented, tested
	Matrix<T> tri_up(bool include_diag = false) const; // Implemented, tested
	T highest_eigenval_iteratedPower(std::vector<T> const& x0, T precision, unsigned long long maxiter) const; // Implemented, tested
	T lowest_eigenval_invIteratedPower(std::vector<T> const& x0, T precision, unsigned long long maxiter) const; // Implemented, tested
	T norm() const; // Implemented
	Matrix<T> adj() const; // Implemented, tested
	T cofactor(unsigned int const line, unsigned int const col) const; // Implemented, tested
	Matrix<T> abs() const; // Implemented, tested
	Matrix<T> eigenvects() const;
	std::vector<T> eigenvals() const;
	T rank() const;
	T trace() const; // Implemented
	Matrix<T> pivot() const; // Implemented
		
	// DECOMPOSITIONS
	std::tuple<Matrix<T>, Matrix<T>> decomp_QR() const;
	std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> decomp_SVD() const;
	std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> decomp_PLU() const; // Implemented, tested
	std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> diagonalize() const;
	Matrix<T> decomp_cholesky() const; // Implemented
	
	// OTHERS
	template<class T2> auto dot(Matrix<T2> const& other) const;  // Implemented, tested
	template<class T2> auto dot(Matrix<T> const&& other) const;
	bool isDiagonal() const; // Implemented
	

	// GENERATORS
	static Matrix<T> gen_random(size_t size, T min, T max); // Implemented
	static Matrix<T> gen_random(size_t lines, size_t cols, T min, T max); // Implemented
	static Matrix<T> gen_diag(size_t size, T value = T()); // Implemented
	static Matrix<T> gen_diag(size_t lines, size_t cols, T value = T()); // Implemented
	static Matrix<T> gen_diag(std::vector<T> values); // Implemented
	static Matrix<T> gen_full(size_t size, T value = T()); // Implemented
	static Matrix<T> gen_full(size_t lines, size_t cols, T value = T()); // Implemented
	static Matrix<T> gen_col(std::vector<T> values); // Implemented
	static Matrix<T> gen_line(std::vector<T> values); // Implemented

	// COMPARATORS
	bool allclose(Matrix<T> other, T abs_precision, T rel_precision) const; // Implemented
	static bool close(T lhs, T rhs, T abs_precision, T rel_precision); // Implemented
	static bool allclose(std::vector<T> const& lhs, std::vector<T> const& rhs, T abs_precision, T rel_precision); // Implemented

	// SOLVERS
	static std::vector<T> solve_descent_col(Matrix<T> const& A, std::vector<T> const& B); // Implemented, NEEDS OPTIMIZATION (MOVE INTO SOLVE_DESCENT ?)
	static std::vector<T> solve_climb_col(Matrix<T> const& A, std::vector<T> const& B); // Implemented, NEEDS OPTIMIZATION (MOVE INTO SOLVE_CLIMB ?)
	static Matrix<T> solve_descent(Matrix<T> const& A, Matrix<T> const& B); // Implemented
	static Matrix<T> solve_climb(Matrix<T> const& A, Matrix<T> const& B); // Implemented
	static Matrix<T> solve_LU(Matrix<T> const& A, Matrix<T> const& B); // Implemented
	static Matrix<T> solve_gaussSeidel(Matrix<T> const& A, Matrix<T> const& B);
	static Matrix<T> solve_jacobi(Matrix<T> const& A, Matrix<T> const& B);
	static Matrix<T> solve_richardson(Matrix<T> const& A, Matrix<T> const& B);
	static Matrix<T> solve_sor(Matrix<T> const& A, Matrix<T> const& B, T const& omega);
	static Matrix<T> solve_gmres(Matrix<T> const& A, Matrix<T> const& B);
	static std::vector<std::tuple<T, T, T>> linearRegression(Matrix<T> const& A, Matrix<T> const& B);

	// MISC
	static void run_tests(); // Implemented
};

#endif