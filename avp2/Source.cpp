
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>


#pragma intrinsic(__rdtsc)
using namespace std;

int getWidthAndHeightInFloat() {
	int widthAndHeight, multipleOf64, widthAndHeightInFloat;
	widthAndHeight = sqrt(6 / 3 * 0.9 * 1024 * 1024);
	multipleOf64 = widthAndHeight / 64;
	widthAndHeightInFloat = multipleOf64 * 64 / 4;
	return widthAndHeightInFloat;
}

int widthAndHeightInFloat = getWidthAndHeightInFloat();
#define line widthAndHeightInFloat
#define matSize line*10


void freeMatrix(float** matrix) {

	for (int i = 0; i < matSize; i++) {

		delete[] matrix[i];
	}
	delete[] matrix;
}

float **createMatrix(bool fillWithZeros) {
	float** matrix = nullptr;
	matrix = new float*[matSize];

	for (int i = 0; i < matSize; i++) {

		matrix[i] = new float[matSize];

		for (int j = 0; j < matSize; j++) {

			float randFloat = 0.0;
			if (!fillWithZeros) {
				randFloat = (float)(rand() % 5);
			}

			matrix[i][j] = randFloat;
		}
	}
	return matrix;
}

void show_matrix(float** matrix) {

	for (int i = 0; i < matSize; i++) {

		for (int j = 0; j < matSize; j++) {

			cout << matrix[i][j] << " ";
		}
		cout << "\n";
	}
}

void multiplyAVX(float** matrixA, float** matrixB, float** resultMatrix) {


	std::chrono::high_resolution_clock::time_point startPoint =
		std::chrono::high_resolution_clock::now();


	for (int i = 0; i < matSize; i++) {

		for (int k = 0; k < matSize; k++) {

			__m256 reg_a_256 = _mm256_set1_ps(matrixA[i][k]);

			for (int j = 0; j < matSize; j += 8) {

				float* matA = matrixB[k] + j;
				float* res = resultMatrix[i] + j;

				__m256 reg_b_256 = _mm256_load_ps(matA);
				__m256 reg_c_256 = _mm256_load_ps(res);

				__m256 reg_mul_256 = _mm256_mul_ps(reg_b_256, reg_a_256);
				reg_c_256 = _mm256_add_ps(reg_mul_256, reg_c_256);

				_mm256_store_ps(res, reg_c_256);
			}
		}
	}
	/*for (int i = 0; i < len; i++) {

		for (int j = 0; j < len; j++) {

			for (int k = 0; k < len; k++) {

				resultMatrix[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}*/


	std::chrono::high_resolution_clock::time_point endPoint =
		std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration =
		std::chrono::duration_cast<std::chrono::duration<double>>(endPoint - startPoint);

	cout << "\nTime : " << duration.count() << " sec\n";

}


void cacheMul(float** a, float** b, float **c) {

	std::chrono::high_resolution_clock::time_point startPoint =
		std::chrono::high_resolution_clock::now();
	//float *rres, *rmul1, *rmul2;
	for (int i = 0; i < matSize; i += line) {
		for (int j = 0; j < matSize; j += line) {
			for (int k = 0; k < matSize; k += line) {
				float* rres = &c[i][j], *mul1 = &a[i][k];
				for (int i2 = 0; i2 < line; i2++ /*, rres += len, rmul1 += len*/) {
					float *rmul2 = &b[k][j];
					mul1 = &a[i + i2][k];
					rres = &c[i + i2][j];
					for (int k2 = 0; k2 < line; k2++/*, rmul2 += 336*/) {
						__m256 a_element = _mm256_set1_ps(mul1[k2]);
						//float a = rmul1[k2];
						rmul2 = &b[k + k2][j];
						for (int j2 = 0; j2 < line; j2 += 8) {
							float* mul2 = &rmul2[j2];
							float* res = &rres[j2];

							__m256 reg_b_256 = _mm256_load_ps(mul2);
							__m256 reg_c_256 = _mm256_load_ps(res);

							__m256 reg_mul_256 = _mm256_mul_ps(reg_b_256, a_element);
							reg_c_256 = _mm256_add_ps(reg_mul_256, reg_c_256);

							_mm256_store_ps(res, reg_c_256);
						}
					}
				}
			}
		}
	}

	std::chrono::high_resolution_clock::time_point endPoint =
		std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration =
		std::chrono::duration_cast<std::chrono::duration<double>>(endPoint - startPoint);

	cout << "Time : " << duration.count() << " sec\n";
	//printMatrix(c);

}


bool CompareMatrices(float** first_matrix, float** second_matrix)
{
	for (int row = 0; row < matSize; row++)
	{
		for (int column = 0; column < matSize; column++)
		{
			if (first_matrix[row][column] != second_matrix[row][column])
			{
				return false;
			}
		}
	}

	return true;
}


//void getCacheSize() {
//	int cacheSizePrev = (sqrt(6 / 3 * 0.9 * 1024 * 1024));
//	int cache = (cacheSizePrev / 64);
//	int cacheSize2 = cache * 64 / 4;
//}

int main() {
	float **matrixA, **matrixB, **matrixC1, **matrixC2;
	//getCacheSize();
	matrixA = createMatrix(false);
	matrixB = createMatrix(false);
	matrixC1 = createMatrix(true);
	matrixC2 = createMatrix(true);
	cout << "AVX multiplication" ;
	multiplyAVX(matrixA, matrixB, matrixC2);
	cout << "Cache optimization multiplication" << endl;
	cacheMul(matrixA, matrixB, matrixC1);
	if (CompareMatrices(matrixC1, matrixC2)) {
		cout << "Matricies are equal" << endl;
	}
	else {
		cout << "Matricies are not equal" << endl;
	}

	freeMatrix(matrixA);
	freeMatrix(matrixB);
	freeMatrix(matrixC1);
	freeMatrix(matrixC2);
	system("pause");
	return 0;
}

