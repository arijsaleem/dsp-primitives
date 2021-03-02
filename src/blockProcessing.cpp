#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

#define PI 3.14159265358979323846
// function for computing the impulse response (reuse from previous experiment)
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

// function for computing the impulse response (reuse from previous experiment)
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

void convolveFIR_block(std::vector<float> &y, const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &f_state, int position)
{
	// allocate memory for the output (filtered) data
	y.resize(x.size(), 0.0);
	float sum = 0.0;

	// block processing
	std::vector<float> x_comb = x;
	x_comb.insert(x_comb.begin(), f_state.begin(), f_state.end());	//concatenate initial state with input vector

	for (auto i = position; i < position+x.size(); i++) {
		sum = 0.0;
		for (auto j = 0; j < h.size(); j++) {
			if ((i - j > 0) && ((i-j-position+h.size()-1)<=x_comb.size()))
				sum += h[j]*x_comb[i-j-position+h.size()-1];
		}

		y[i-position] = sum;
	}

	//updating the f_state (zi)
	std::vector<float> zf(&x[x.size()-h.size()+1], &x[x.size()]);
	f_state = zf;
}

// function to read audio data from a binary file that contains raw samples
// represented as 32-bit floats; we also assume two audio channels
// note: check the Python script that can prepare this type of files
// directly from .wav files
void read_audio_data(const std::string in_fname, std::vector<float> &audio_data)
{
	// file descriptor for the input to be read
	std::ifstream fdin(in_fname, std::ios::binary);
	if(!fdin) {
		std::cout << "File " << in_fname << " not found ... exiting\n";
		exit(1);
	} else {
		std::cout << "Reading raw audio from \"" << in_fname << "\"\n";
	}
	// search for end of file to count the number of samples to be read
	fdin.seekg(0, std::ios::end);
	// we assume the Python script has written data in 32-bit floats
	const unsigned int num_samples = fdin.tellg() / sizeof(float);

	// allocate memory space to store all the samples
	audio_data.resize(num_samples);
	// back to the beginning of the file to read all samples at once
	fdin.seekg(0, std::ios::beg);
	// do a single read for audio data from the input file stream
	fdin.read(reinterpret_cast<char*>(&audio_data[0]), \
						num_samples*sizeof(float));
	// close the input file
	fdin.close();
}

void filter_block_processing(std::vector<float> &audio_left, std::vector<float> &audio_right, const unsigned int block_size, std::vector<float> &h, std::vector<float> &output_left, std::vector<float> &output_right)
{
	// output_left.resize(audio_left.size()+h.size()-1, 0.0);
	// output_right.resize(audio_right.size()+h.size()-1, 0.0);

	// state
	std::vector<float> f_state_l(h.size()-1);
	std::vector<float> f_state_r(h.size()-1);

	int position = 0;
	int cnt = 0;

	while(1) {
		std::vector<float> block_left(&audio_left[position], &audio_left[position+block_size]);
		std::vector<float> filtered_l(block_size);
		convolveFIR_block(filtered_l, block_left, h, f_state_l, position);
		output_left.insert(output_left.end(), filtered_l.begin(), filtered_l.end());

		std::vector<float> block_right(&audio_right[position], &audio_right[position+block_size]);
		std::vector<float> filtered_r(block_size);
		convolveFIR_block(filtered_r, block_right, h, f_state_r, position);
		output_right.insert(output_right.end(), filtered_r.begin(), filtered_r.end());

		position += block_size;

		if (position > audio_left.size())
			break;
	}
}

// function to split an audio data where the left channel is in even samples
// and the right channel is in odd samples
void split_audio_into_channels(const std::vector<float> &audio_data, std::vector<float> &audio_left, std::vector<float> &audio_right)
{
	for (auto i=0; i<audio_data.size(); i++) {
		if (i%2 == 0)
			audio_left.push_back(audio_data[i]);
		else
			audio_right.push_back(audio_data[i]);
	}
}

// function to write audio data to a binary file that contains raw samples
// represented as 32-bit floats; we also assume two audio channels
// note: check the python script that can read this type of files
// and then reformat them to .wav files to be run on third-party players
void write_audio_data(const std::string out_fname, const std::vector<float> &audio_left, const std::vector<float> &audio_right)
{
	// file descriptor for the output to be written
	if (audio_left.size() != audio_right.size()) {
		std::cout << "Something got messed up with audio channels\n";
		std::cout << "They must have the same size ... exiting\n";
		exit(1);
	} else {
		std::cout << "Writing raw audio to \"" << out_fname << "\"\n";
	}
	std::ofstream fdout(out_fname, std::ios::binary);
	for (auto i=0; i<audio_left.size(); i++) {
		// we assume we have handled a stereo audio file
		// hence, we must interleave the two channels
		// (change as needed if testing with mono files)
		fdout.write(reinterpret_cast<const char*>(&audio_left[i]),\
								sizeof(audio_left[i]));
		fdout.write(reinterpret_cast<const char*>(&audio_right[i]),\
								sizeof(audio_right[i]));
	}
	fdout.close();
}

int main()
{
	// assume the wavio.py script was run beforehand to produce a binary file
	const std::string in_fname = "../data/float32samples.bin";
	// declare vector where the audio data will be stored
	std::vector<float> audio_data;
	// note: we allocate memory for audio_data from within this read function
	read_audio_data(in_fname, audio_data);

	// set up the filtering flow
	float Fs = 44100.0;	// sample rate for our "assumed" audio (change as needed for 48 ksamples/sec audio files)
	float Fc = 1000.0;	// cutoff frequency (explore ... but up-to Nyquist only!)
	// number of FIR filter taps (feel free to explore ...)
	unsigned short int num_taps = 491;
	unsigned short int block_size = 500;

	// impulse response (reuse code from the previous experiment)
	std::vector<float> h;
	impulseResponseLPF(Fs, Fc, num_taps, h);
	// note: memory for the impulse response vector and output data vectors
	// should be allocated from within the corresponding functions
	// (as for the previous experiment, from where you should reuse your code)

	// there is one more point before filtering is done:
	// recall we assume there are two channels in the audio data
	// the channels must be handled separately by your DSP functions, hence
	// split the audio_data into two channels (audio_left and audio_right)

	// declare vectors where the audio left/right channels will be stored
	std::vector<float> audio_left, audio_right;
	// note: we allocate the memory for the left/right channels
	// from within the split function that is called in the code below
	split_audio_into_channels(audio_data, audio_left, audio_right);

	//do block processing here
	std::vector<float> block_processing_left, block_processing_right;
	filter_block_processing(audio_left, audio_right, block_size, h, block_processing_left, block_processing_right);

	// convolution code for filtering (reuse from the previous experiment)
	// std::vector<float> single_pass_left, single_pass_right;
	// convolveFIR(single_pass_left, audio_left, h);
	// convolveFIR(single_pass_right, audio_right, h);
	// note: by default the above convolution produces zero on the output stream
	// YOU will need to update the convolveFIR and impulseResponseLPF functions

	// create a binary file to be read by wavio.py script to produce a .wav file
	// note: small adjustments will need to be made to wavio.py, i.e., you should
	// match the filenames, no need for self-checks as default Python code, ...
	const std::string out_fname = "../data/float32filtered.bin";

	//single pass
	// write_audio_data(out_fname, single_pass_left,	single_pass_right);

	//block processing
	write_audio_data(out_fname, block_processing_left,	block_processing_right);

	return 0;
}
