#include "matrix.h"
#include "matrix_test.h"

int main() {
	// The default executable just runs all the tests for the long double type
	Matrix_test<long double>::run_all();
	return 0;
}