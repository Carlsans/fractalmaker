#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <Windows.h>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "matrix.h"
#include "pybind_matrix.h"
#include <stack> 
#include <cassert>

using namespace std;

	

const double e = 2.7182818284590452353602874713527;
namespace py = pybind11;

double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double tanh_impl(double x) {
    return sinh_impl(x) / cosh_impl(x);
} 

int sum_array(int myarray[], int length) {
	int mysum = 0;
	for (int i = 0; i < length;i++) {
		mysum += myarray[i];
	
	}
	return mysum;
}

int mandelbrot(double real, double imag) {
	int limit = 100;
	double zReal = real;
	double zImag = imag;

	for (int i = 0; i < limit; ++i) {
		double r2 = zReal * zReal;
		double i2 = zImag * zImag;

		if (r2 + i2 > 4.0) return i;

		zImag = 2.0 * zReal * zImag + imag;
		zReal = r2 - i2 + real+0;
	}
	return limit;
}


class Matrix2 {
public:
	Matrix2(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
		m_data = new int[rows * cols];
	}
	int* data() { return m_data; }
	size_t rows() const { return m_rows; }
	size_t cols() const { return m_cols; }
private:
	size_t m_rows, m_cols;
	int* m_data;
};



Matrix<double> mul(const Matrix<double>& A, const Matrix<double>& B)
{
	if (A.shape() != B.shape())
		throw std::length_error("Matrix 'A' and 'B' are inconsistent");

	Matrix<double> ret(A.shape());

	for (int i = 0; i < A.size(); i++)
		ret[i] = A[i] * B[i];

	return ret;
}

Matrix <int> test(Matrix <int>& test) {
	// Create a vector containing integers
	//std::vector<int> matrixsize = { 100, 100};
	Matrix<int> anothermatrix({10, 10});
	//returnmatrix
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			anothermatrix(i, j) = i*j;
			test(i,j) = i + j;
		}
	}
	std::cout << test.ndim() << "\n";
	std::cout << test.size() << "\n";
	//std::cout << test.shape() << "\n";
	std::cout << test(1,1) << "\n";
	test(1,1) = 22;
	//Matrix<int> returnmatrix[2][2][2];
	//std::cout << "static constructor\n";
	return anothermatrix;
}
Matrix<unsigned char> mandelbrotmatrix(unsigned long xwidth, unsigned long ywidth, float xmin, float xmax, float ymin, float ymax) {
	float xrange = xmax - xmin;
	float yrange = ymax - ymin;
	float xstep = xrange / xwidth;
	float ystep = yrange / ywidth;
	float creal;
	float cimag;
	Matrix<unsigned char> fractalmatrix({ xwidth, ywidth });
	for (int i = 0; i < xwidth; i++) {
		for (int j = 0; j < ywidth; j++) {
			creal = xmin + i * xstep;
			cimag = ymin + j * ystep;
			fractalmatrix(i, j) = mandelbrot(creal,cimag);


		}


	}

	return fractalmatrix;
}
template <class T>
complex<T> variablefct(unsigned char fctindex, complex<T> c, complex<T> z) {
	switch (fctindex) {
	case 1:
		return c;
		break;
	case 2:
		return -c;
		break;
	case 3:
		return z;
		break;
	case 4:
		return -z;
		break;
	case 5:
		return c + z;
		break;
	case 6:
		return c - z;
		break;
	case 7:
		return c * z;
		break;
	case 8:
		if (z.real() == 0 && z.imag() == 0) return 1;
		return c / z;
		break;
	case 9:
		return -c * z;
		break;
	case 10:
		if (z.real() == 0 && z.imag() == 0) return 1;
		return -c / z;
		break;
	case 11:
		return pow(c , z);
		break;
	case 12:
		return pow(z, c);
		break;
	case 13:
		return pow(c, -z);
		break;
	case 14:
		return pow(z, -c);
		break;
	case 15:
		return z * z;
		break;
	case 16:
		return -(z * z);
		break;
	case 17:
		return c * c;
		break;
	case 18:
		return -(c * c);
		break;
	case 19:
		return (-z * -z);
		break;
	case 20:
		return -(-z*-z);
		break;
	case 21:
		return (-c*-c);
		break;
	case 22:
		return -(-c*-c);
		break;
	default:
		std::cout << "fct index wrong in variable";
		return 0;
	}

}
template <class T>
complex<T> functionfct(unsigned char fctindex,complex <T> zvalue) {
	switch (fctindex) {
	case 1:
		return sin(zvalue);
		break;
	case 2:
		return cos(zvalue);
		break;
	case 3:
		return tan(zvalue);
		break;
	case 4:
		return asin(zvalue);
		break;
	case 5:
		return acos(zvalue);
		break;
	case 6:
		return atan(zvalue);
		break;
	case 7:
		return sinh(zvalue);
		break;
	case 8:
		return cosh(zvalue);
		break;
	case 9:
		return tanh(zvalue);
		break;
	case 10:
		return asinh(zvalue);
		break;
	case 11:
		return acosh(zvalue);
		break;
	case 12:
		return atanh(zvalue);
		break;
	case 13:
		return log(zvalue);
		break;
	case 14:
		return log10(zvalue);
		break;
	case 15:
		return exp(zvalue);
		break;
	case 16 :
		return sqrt(zvalue);
		break;
	case 17:
		return abs(zvalue);
		break;
	case 18:
		return conj(zvalue);
		break;
	case 19:
		return zvalue*zvalue;
		break;
	case 20:
		return zvalue;
		break;

	default:
		std::cout << "fct index wrong in functions";
		return 0;
	}

}
bool isoperatorpriority(unsigned char fctindex) {
	if (fctindex == 1 || fctindex == 2) {
		return false;
	}
	else {
		return true;
	}

}
template <class T>
bool iscomplexzero(complex<T>complexnumber) {
	if (complexnumber.real() == 0 && complexnumber.imag() == 0) {
		return true;
	}
	return false;
}
template <class T>
complex<T> operatorfct(unsigned char fctindex, complex <T> tempz, complex <T> z) {
	if (iscomplexzero<T>(tempz)) {
		switch (fctindex) {
		case 1:
			return z + tempz;
			break;
		case 2:
			return z - tempz;
			break;
		case 3:
			return z + tempz;
			break;
		case 4:
			return z - tempz;
			break;
		case 5:
			return -1;

		default:
			std::cout << "fct index wrong in functions";
			return 0;
		}
	}
	else {
		switch (fctindex) {
		case 1:
			return z + tempz;
			break;
		case 2:
			return z - tempz;
			break;
		case 3:
			return z * tempz;
			break;
		case 4:
			if (tempz.real() == 0 && tempz.imag() == 0) return 1;
			return z / tempz;
			break;
		case 5:
			return pow(z, tempz);

		default:
			std::cout << "fct index wrong in functions";
			return 0;
		}
	}
	

}

template <class T>
unsigned char calculateformula(complex<T> c, Matrix <int>& formulamatrix, Matrix <T>& coefficientmatrix,int limit) {
	int formulalinesnumber = formulamatrix.shape(0)[0];
	//int limit = 100;
	//complex <T> z = (0, 0);
	complex <T> lastz = (0, 0);
	complex <T> tempz = (0, 0);
	complex <T> buffer = (0, 0);
	//complex <T> operatorstoringvar = (0, 0);
	stack <complex <T>> zstack;
	//zstack.push(tempz);
	for (int j = 0; j < limit; j++) {
		//operatorstoringvar = (0, 0);
		for (int i = 0; i < formulalinesnumber; i++) {
			tempz = (0, 0);
			tempz = variablefct<T>(formulamatrix(i, 0), c, lastz) * coefficientmatrix(i, 0) + coefficientmatrix(i, 2);
			tempz = functionfct<T>(formulamatrix(i, 1), tempz) * coefficientmatrix(i, 1) + coefficientmatrix(i, 3);
if (i != formulalinesnumber - 1 || i != 0) {
	if (isoperatorpriority(formulamatrix(i, 2))) {
		buffer = operatorfct(formulamatrix(i, 2), tempz, buffer);
	}
	else {
		zstack.push(buffer);
		buffer = operatorfct<T>(formulamatrix(i, 2), tempz, complex<T>(0, 0));
	}
}
else if (i == 0) {
	buffer = operatorfct(formulamatrix(i, 2), tempz, complex<T>(0, 0));
}//end of i
else {
	buffer = operatorfct(formulamatrix(i, 2), tempz, buffer);
	zstack.push(buffer);
}
		}
		lastz = (0, 0);
		while (!zstack.empty()) {
			lastz = lastz + zstack.top();
			zstack.pop();
		}
		//lastz = z;
		if (std::norm(lastz) > 4.0) return j;
	}
	return limit;


}
/*
unsigned char calculateformula(complex<float> c, Matrix <int>& formulamatrix) {
	int formulalinesnumber = formulamatrix.shape(0)[0];
	int limit = 100;
	complex <float> z = (0,0);
	complex <float> lastz = (0, 0);
	complex <float> tempz = (0, 0);
	complex <float> operatorstoringvar = (0,0);
	stack <complex <float>> stack;
	stack.push(tempz);
	for (int j = 0; j < limit; j++) {
		operatorstoringvar = (0, 0);
		for (int i = 0; i < formulalinesnumber; i++) {
			tempz = (0, 0);
			tempz = variablefct(formulamatrix(i, 0), c, lastz);
			tempz = functionfct(formulamatrix(i, 1), tempz);
			if (i != formulalinesnumber - 1 || i != 0) {
				if (isoperatorpriority(formulamatrix(i, 2))) {
					if (iscomplexzero(operatorstoringvar)) {
						operatorstoringvar = operatorfct(formulamatrix(i, 2), tempz, z);
					}
					else {
						operatorstoringvar = operatorfct(formulamatrix(i, 2), tempz, operatorstoringvar);
					}

				}
				else if (iscomplexzero(operatorstoringvar)) {
					z = operatorfct(formulamatrix(i, 2), tempz, z);
				}
				else {
					z = operatorfct(formulamatrix(i, 2), tempz, operatorstoringvar);
					operatorstoringvar = (0, 0);

				}

			}
			else if(iscomplexzero(operatorstoringvar)) {
				z = operatorfct(formulamatrix(i, 2), tempz, z);
			}
			else {
				z = operatorfct(formulamatrix(i, 2), tempz, operatorstoringvar);
			}

		}
		lastz = z;
		if (std::norm(z) > 4.0) return j;
	}
	return limit;


}
*/
/*
Matrix<unsigned char> fractalmatrix(unsigned long xwidth, unsigned long ywidth, float xmin, float xmax, float ymin, float ymax, Matrix <int>& formulamatrix, Matrix <float>& coefficientmatrix) {
	float xrange = xmax - xmin;
	float yrange = ymax - ymin;
	float xstep = xrange / xwidth;
	float ystep = yrange / ywidth;
	float creal;
	float cimag;
	float mantissa;
	complex<float> c;
	//std::cout << formulalinesnumber << '\n';
	//std::cout << formulamatrix.shape(0)[1] << '\n';
	//std::lerp(5.0, 10.0, t)

	// Deinterpolate coefficients
	int coefflinesnumber = coefficientmatrix.shape(0)[0];
	int coeffcolumnsnumber = 4;
	for (int i = 0; i < coefflinesnumber; i++) {
		for (int j = 0; j < coeffcolumnsnumber; j++)
		{
			//coefficientmatrix(i, j) -= .5;
			
			//coefficientmatrix(i, j) = coefficientmatrix(i, j) * 2;
			//coefficientmatrix(i, j) = pow(coefficientmatrix(i, j),2)*100;
			
			//cout << coefficientmatrix(i, j) << '\n';
			
			//coefficientmatrix(i, j) = abs(pow((coefficientmatrix(i, j) - 0.5), 3)* 100);
			coefficientmatrix(i, j) = (coefficientmatrix(i, j) - .5) * 2;
			//cout << coefficientmatrix(i, j) << '\n';
			//coefficientmatrix(i, j) = ldexp((coefficientmatrix(i, j) - 0.5), 3);

			
		}
	}
	
	//std::cout << "ln" << coefflinesnumber << "col" << coeffcolumnsnumber <<'\n';
	Matrix<unsigned char> fractalmatrix({ xwidth, ywidth });
	for (int i = 0; i < xwidth; i++) {
		for (int j = 0; j < ywidth; j++) {
			c.real(xmin + i * xstep);
			c.imag(ymin + j * ystep);
			//c = c * coefficientmatrix(i, 0) + coefficientmatrix(i, 2);
			fractalmatrix(i, j) = calculateformula(c, formulamatrix, coefficientmatrix);


		}


	}

	return fractalmatrix;
}
*/
template <class T>
Matrix<unsigned char> fractalmatrixtemplate(unsigned long xwidth, unsigned long ywidth, T xmin, T xmax, T ymin, T ymax, Matrix <int>& formulamatrix, Matrix <T>& coefficientmatrix,int iterations) {
	T xrange = xmax - xmin;
	T yrange = ymax - ymin;
	T xstep = xrange / xwidth;
	T ystep = yrange / ywidth;
	T creal;
	T cimag;
	T mantissa;
	complex<T> c;
	//std::cout << formulalinesnumber << '\n';
	//std::cout << formulamatrix.shape(0)[1] << '\n';
	//std::lerp(5.0, 10.0, t)

	// Deinterpolate coefficients
	int coefflinesnumber = coefficientmatrix.shape(0)[0];
	int coeffcolumnsnumber = 4;
	for (int i = 0; i < coefflinesnumber; i++) {
		for (int j = 0; j < coeffcolumnsnumber; j++)
		{
			//coefficientmatrix(i, j) -= .5;

			//coefficientmatrix(i, j) = coefficientmatrix(i, j) * 2;
			//coefficientmatrix(i, j) = pow(coefficientmatrix(i, j),2)*100;

			//cout << coefficientmatrix(i, j) << '\n';
			/*unsigned int* ptr = (unsigned int*)&coefficientmatrix(i, j);
			unsigned int s = *ptr >> 31;
			unsigned int e = *ptr & 0x7f800000;
			e = e >> 23;
			unsigned int m = *ptr & 0x007fffff;
			cout << "s=" << s << "e=" << e << "m" << m << "num" << coefficientmatrix(i, j) << '\n';*/
			//coefficientmatrix(i, j) = abs(pow((coefficientmatrix(i, j) - 0.5), 3)* 100);
			coefficientmatrix(i, j) = (coefficientmatrix(i, j) - .5) * 2;
			//cout << coefficientmatrix(i, j) << '\n';
			//coefficientmatrix(i, j) = ldexp((coefficientmatrix(i, j) - 0.5), 3);


		}
	}
	
	
	
	//std::cout << "ln" << coefflinesnumber << "col" << coeffcolumnsnumber <<'\n';
	Matrix<unsigned char> fractalmatrix({ xwidth, ywidth });
	for (int i = 0; i < xwidth; i++) {
		for (int j = 0; j < ywidth; j++) {
			c.real(xmin + i * xstep);
			c.imag(ymin + j * ystep);
			//c = c * coefficientmatrix(i, 0) + coefficientmatrix(i, 2);
			fractalmatrix(i, j) = calculateformula(c, formulamatrix, coefficientmatrix,iterations);


		}


	}

	return fractalmatrix;
}

Matrix<unsigned char> fractalmatrix(unsigned long xwidth, unsigned long ywidth, float xmin, float xmax, float ymin, float ymax, Matrix <int>& formulamatrix, Matrix <float>& coefficientmatrix, int iterations) {
	return fractalmatrixtemplate<float>(xwidth, ywidth, xmin, xmax, ymin, ymax, formulamatrix, coefficientmatrix, iterations);
}

Matrix<unsigned char> fractalmatrixdouble(unsigned long xwidth, unsigned long ywidth, double xmin, double xmax, double ymin, double ymax, Matrix <int>& formulamatrix, Matrix <double>& coefficientmatrix, int iterations) {
	return fractalmatrixtemplate<double>(xwidth, ywidth, xmin, xmax, ymin, ymax, formulamatrix, coefficientmatrix, iterations);
}

PYBIND11_MODULE(superfastcode, m) {
py::class_<Matrix2>(m, "Matrix", py::buffer_protocol())
.def_buffer([](Matrix2& m) -> py::buffer_info {
	return py::buffer_info(
		m.data(),                               /* Pointer to buffer */
		sizeof(int),                          /* Size of one scalar */
		py::format_descriptor<int>::format(), /* Python struct-style format descriptor */
		2,                                      /* Number of dimensions */
		{ m.rows(), m.cols() },                 /* Buffer dimensions */
		{ sizeof(int) * m.cols(),             /* Strides (in bytes) for each index */
		  sizeof(int) }
	);
	});
m.def("mandelbrotmatrix", &mandelbrotmatrix, "Mandelbrot matrix",
	py::arg("xwidth"), py::arg("ywidth"), py::arg("xmin"), py::arg("xmax"),py::arg("ymin"), py::arg("ymax"));
m.def("fractalmatrix", &fractalmatrix, "Fractal matrix",
	py::arg("xwidth"), py::arg("ywidth"), py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"), py::arg("formulamatrix"), py::arg("coefficientmatrix"), py::arg("iterations"));
m.def("fractalmatrixdouble", &fractalmatrixdouble, "Fractal matrix double",
	py::arg("xwidth"), py::arg("ywidth"), py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"), py::arg("formulamatrix"), py::arg("coefficientmatrix"), py::arg("iterations"));
m.def("test", &test, "Test");
m.def("mul", &mul, "Multiplication");
#ifdef VERSION_INFO
m.attr("__version__") = VERSION_INFO;
#else
m.attr("__version__") = "dev";
#endif
}
/*
PYBIND11_MODULE(superfastcode, m) {
    m.def("fast_tanh2", &tanh_impl, R"pbdoc(
        Compute a hyperbolic tangent of a single argument expressed in radians.
    )pbdoc");
    m.def("mandelbrot", &mandelbrot, "Mandelbrot iteration",
        py::arg("real"), py::arg("imag"));
	m.def("sum_array", &sum_array, "Sum array content (int only)",
		py::arg("array"), py::arg("content"));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}*/

