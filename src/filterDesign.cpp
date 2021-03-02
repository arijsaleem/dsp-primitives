#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>

#define PI 3.14159265358979323846

// function for DFT (reused from previous experiment)
void DFT(const std::vector<float> &x, std::vector<std::complex<float>> &Xf)
{
	Xf.resize(x.size(), static_cast<std::complex<float>>(0, 0));
	for (auto m = 0; m < Xf.size(); m++) {
		for (auto k = 0; k < x.size(); k++) {
				std::complex<float> expval(0, -2*PI*(k*m) / x.size());
				Xf[m] += x[k] * std::exp(expval);
		}
	}
}

void smallDFT(const std::vector<float> &x, std::vector<std::complex<float>> &Xf)
{
	unsigned int size = 1024;
	unsigned int pos = 0;
	// std::vector<float> samples(&x[0], &x[size]);
	Xf.resize(size / 4, static_cast<std::complex<float>>(0, 0));

	for (auto i = 0; i < 4; i++) {
		std::vector<float> samples(&x[pos], &x[pos+256]);
		std::vector<std::complex<float>> Sf;
		DFT(samples, Sf);
		std::transform (Xf.begin(), Xf.end(), Sf.begin(), Xf.begin(), std::plus<std::complex<float>>());
		pos += 256;
		// std::cout << "Sf at index 1 " << Sf[2] << "\n";
	}
	std::complex<float> m = 4;
	for (auto j = 0; j < Xf.size(); j++) {
		Xf[j] = Xf[j] / m;
	}

}

// function to generate a sine with N samples per second over interval
void generateSin(std::vector<float> &t, std::vector<float> &x, float Fs, float interval, float frequency = 7.0, float amplitude = 5.0, float phase = 0.0)
{
	// we do NOT allocate memory space explicitly
	// for the time (t) vector and sample (x) vector
	t.resize(0); x.resize(0);
	float dt = 1/Fs;
	for (auto i = 0.0; i < interval; i += dt) {
		// vector size increases when pushing new elements into it
		t.push_back(i);
		x.push_back(amplitude*std::sin(2*PI*frequency*i+phase));
	}
}

// function to mix an array of sines
void mixSin(const std::vector<std::vector<float>> &sv, std::vector<float> &mixed)
{
	// assumes at least one sine passed
	// assumes all input sines are of the same size
	for (auto i = 0.0; i < sv[0].size(); i ++) {
		float mixval = 0.0;
		// note: sv.size() returns the number of sines (or rows in 2D repr)
		// sv[0].size() returns the number of samples in a sine (or cols in 2D repr)
		for (auto k = 0; k < sv.size(); k++)
			mixval += sv[k][i];
		mixed.push_back(mixval);
	}
}

// function to print a real vector (reused from previous experiment)
void printRealVector(const std::vector<float> &x)
{
	std::cout << "Printing float vector of size " << x.size() << "\n";
	for (auto i = 0; i < x.size(); i++)
		std::cout << x[i] << " ";
	std::cout << "\n";
}

// function to print a complex vector (reused from previous experiment)
void printComplexlVector(const std::vector<std::complex<float>> &X)
{
	std::cout << "Printing complex vector of size " << X.size() << "\n";
	for (auto i = 0; i < X.size(); i++)
		std::cout << X[i] << " ";
	std::cout << "\n";
}

// function to record data in a format to be read by GNU plot
// the arguments are VERY specific to this usage in this experiment
// we have the time vector (t), a vector of sines (sv),
// input samples (x, i.e., mixed sines for this experiment),
// output samples (y, filtered input samples)
// frequency vectors (both Xf and Yf)
// note: the reference code does NOT do filtering, hence y and Yf are zeros by default
void plotMixedSinesSpectrum(const std::vector<float> &t, const std::vector<std::vector<float>> &sv, const std::vector<float> &x, const std::vector<float> &y, const std::vector<std::complex<float>> &Xf, const std::vector<std::complex<float>> &Yf)
{
	// write data in text format to be parsed by gnuplot
	const std::string filename = "../data/example.dat";
	std::fstream fd;  // file descriptor
	fd.open(filename, std::ios::out);
	fd << "#\tindex\tsine(0)\tsine(1)\tsine(2)\tdata in\tdata out\tspectrum in\tspectrum out\n";
	for (auto i = 0; i < t.size(); i++) {
		fd << "\t " << i << "\t";
		for (auto k = 0; k < sv.size(); k++)
			fd << std::fixed << std::setprecision(3) << sv[k][i] << "\t ";
	
		fd << x[i] << "\t "<< y[i] << "\t\t ";
		fd << std::abs(Xf[i])/Xf.size() << "\t\t " << std::abs(Yf[i])/Yf.size() <<"\n";
	}
	std::cout << "Generated " << filename << " to be used by gnuplot\n";
	fd.close();
}

// function to compute the impulse response "h" based on the sinc function
// see pseudocode from previous lab that was implemented in Python
void impulseResponseLPF(float Fs, float Fc, unsigned short int num_taps, std::vector<float> &h)
{
	// allocate memory for the impulse response
	h.resize(num_taps, 0.0);
	//normalized cutoff Frequency
	float norm_cutoff = Fc/(Fs/2);
	float numerator, denominator;

	//derive filter coefficients
	for (auto i = 0; i < num_taps; i++) {
		if (i == ((num_taps-1)/2))
			h[i] = norm_cutoff;
		else {
			numerator = sin(PI*norm_cutoff*(i - ((num_taps-1)/2)));
			denominator = PI*norm_cutoff*(i - ((num_taps-1)/2));
			h[i] = norm_cutoff*(numerator/denominator);
		}
		//apply the Hann window
		h[i] = h[i]*pow(sin(i*PI/num_taps),2);
	}
}

// function to compute the filtered output "y" by doing the convolution
// of the input data "x" with the impulse response "h"; this is based on
// your Python code from the take-home exercise from the previous lab
void convolveFIR(std::vector<float> &y, const std::vector<float> &x, const std::vector<float> &h)
{
	// allocate memory for the output (filtered) data
	y.resize(x.size()+h.size()-1, 0.0);
	float sum;

	// single pass
	for (auto i = 0; i < x.size(); i++) {
		sum = 0.0;
		for (auto j = 0; j < h.size(); j++) {
			if (i - j > 0)
				sum += h[j]*x[i-j];
		}
		y[i] = sum;
	}
}

int main()
{

	float Fs = 1024.0;                    // samples per second
	float interval = 1.0;                 // number of seconds
	unsigned short int num_taps = 101;    // number of filter taps
	float Fc = 200.0;                      // cutoff frequency (in Hz)

	// declare a vector of vectors for multiple sines
	std::vector<std::vector<float>> sv, sv1;
	// declare time and sine vectors
	std::vector<float> t, sine;
	// note: there is no explicit memory allocation through vector resizing
	// vector memory space will increase via the push_back method

	// generate and store the first tone
	// check the function to understand the order of arguments

	// for regular DFT
	generateSin(t, sine, Fs, interval, 10.0, 5.0, 0.0);
	sv.push_back(sine);
	// generate and store the second tone
	generateSin(t, sine, Fs, interval, 40.0, 2.0, 0.0);
	sv.push_back(sine);
	// generate and store the third tone
	generateSin(t, sine, Fs, interval, 50.0, 3.0, 0.0);
	sv.push_back(sine);

	// for smallDFT
	generateSin(t, sine, Fs, interval, 40.0, 5.0, 0.0);
	sv1.push_back(sine);
	// generate and store the second tone
	generateSin(t, sine, Fs, interval, 120.0, 2.0, 0.0);
	sv1.push_back(sine);
	// generate and store the third tone
	generateSin(t, sine, Fs, interval, 240.0, 3.0, 0.0);
	sv1.push_back(sine);

	// declare the mixed sine vector and mix the three tones
	std::vector<float> x, x1;
	mixSin(sv, x);
	mixSin(sv1, x1);
	// printRealVector(x);

	// declare a vector of complex values for DFT; no memory is allocated for it
	std::vector<std::complex<float>> Xf, Xf1;
	// DFT(x, Xf);
	// smallDFT(x1, Xf1);
	// printComplexlVector(Xf);

	// generate the impulse response h
	// convolve it with the input data x
	// in order to produce the output data y
	std::vector<float> h;              // impulse response
	impulseResponseLPF(Fs, Fc, num_taps, h);
	std::vector<float> y, y1;              // filter out
	// convolveFIR(y, x, h);

	// compute DFT of the filtered data
	std::vector<std::complex<float>> Yf, Yf1;
	// DFT(y, Yf);
	// smallDFT(y1, Yf1);

	// prepare the data for gnuplot
	// plotMixedSinesSpectrum(t, sv, x, y, Xf, Yf);

	// std::cout << "Printing float vector of size " << Xf.size() << "\n";
	// for (auto i = 0; i < 10; i++)
	// 	std::cout << Yf[i] << " ";
	//
	// std::cout << "Printing float vector of size " << Xf.size() << "\n";
	// for (auto i = 0;  i< 10; i++)
	// 	std::cout << Xf[i] << " ";

	// Takehome exercise part 2
	auto start_time = std::chrono::high_resolution_clock::now();
	DFT(x, Xf);
	auto stop_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> DFT_run_time = stop_time-start_time;
	std::cout << "DFT ran for " << DFT_run_time.count() << " milliseconds" << "\n";

	start_time = std::chrono::high_resolution_clock::now();
	smallDFT(x1, Xf1);
	stop_time = std::chrono::high_resolution_clock::now();
	DFT_run_time = stop_time-start_time;
	std::cout << "smallDFT ran for " << DFT_run_time.count() << " milliseconds" << "\n";
	// convolveFIR(y1, x1, h);
	// smallDFT(y1, Yf1);
	// plotMixedSinesSpectrum(t, sv1, x1, y1, Yf1, Yf1);

	// naturally, you can comment the line below once you are comfortable to run gnuplot
	std::cout << "Run: gnuplot -e 'set terminal png size 1024,768' example.gnuplot > ../data/example.png\n";

	return 0;
}
