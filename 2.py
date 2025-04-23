import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as signal
font = {'family' : 'Helvetica Neue'}

matplotlib.rc('font', **font)
def compute_fft(signal, sampling_rate=1.0, return_magnitude=True):
    """
    Compute the Fast Fourier Transform of a NumPy array.

    Parameters:
    - signal: NumPy array, input time-domain signal
    - sampling_rate: float, sampling frequency in Hz (default 1.0)
    - return_magnitude: bool, if True returns magnitude spectrum, if False returns complex FFT

    Returns:
    - frequencies: NumPy array of frequency bins
    - fft_result: NumPy array of FFT magnitude (or complex values if return_magnitude=False)
    """

    # Ensure input is a NumPy array
    signal = np.asarray(signal)

    # Number of samples
    n_samples = len(signal)

    # Compute FFT
    fft_output = np.fft.fft(signal)
    # Compute frequency bins
    frequencies = np.fft.fftfreq(n_samples, d=1 / sampling_rate)

    # Only keep positive frequencies (up to Nyquist frequency)
    positive_mask = frequencies >= 0
    frequencies = frequencies[positive_mask]
    fft_output = fft_output[positive_mask]

    if return_magnitude:
        # Compute magnitude spectrum (absolute value)
        fft_result = np.abs(fft_output)
    else:
        # Return complex FFT values
        fft_result = fft_output

    return frequencies, fft_result

def signal_random(n_samples):
    return np.random.normal(size=n_samples)

def signal_imp(n_samples, ramp_size=1, offset=0, ramp_size_decrease=0, one_length=0):
    s = np.zeros(n_samples)
    # make spike with up and down ramp, the ramp starts at "offset"
    # and ends at "offset + ramp_size"
    s[offset:offset + ramp_size] = np.linspace(0, 1, ramp_size)
    s[offset + ramp_size+one_length:offset + ramp_size_decrease + ramp_size+one_length] = np.linspace(1, 0, ramp_size_decrease)
    s[offset+ramp_size:offset+ramp_size+one_length] = np.ones(one_length)

    return s

def create_test_signal(n_samples):
    """Create a test signal with noise"""
    # return signal_imp(n_samples, ramp_size=0, offset=100, ramp_size_decrease=0, one_length=50)
    return signal_random(n_samples)


def apply_convolution(signal, kernel, m_times=1):
    """Apply convolution filter m times"""
    result = signal.copy()
    for _ in range(m_times):
        result = np.convolve(result, kernel, mode='same')
    return result

def compute_fft_sqrt(_signal, n_samples):
    fft = np.fft.fft(_signal)
    u = np.zeros(n_samples)
    for i in range(n_samples):
        u[i] = np.sqrt(fft[i].real ** 2 + fft[i].imag ** 2)
    return u


def plot_signals(original, filtered, kernel, m_times):
    """Plot original and filtered signals"""
    plt.figure(figsize=(10, 10))
    NUMBER_OF_PLOTS = 6

    # Plot original signal
    plt.subplot(NUMBER_OF_PLOTS, 1, 1)
    plt.plot(original, 'b-', label='Original Signal')
    plt.title('Original Signal')
    plt.legend()
    plt.grid(True)

    # Plot original signal FFT
    plt.subplot(NUMBER_OF_PLOTS, 1, 2)
    plt.plot(compute_fft(original,return_magnitude=True)[1], 'b-', label='FFT of Original Signal')
    # plt.stem(compute_fft(original,return_magnitude=True)[1], 'b-', label='FFT of Original Signal')
    plt.title('Original Signal FFT')
    plt.legend()
    plt.grid(True)


    # Plot filtered signal
    plt.subplot(NUMBER_OF_PLOTS, 1, 3)
    plt.plot(filtered, 'r-', label=f'Filtered (m={m_times})')
    plt.title('Filtered Signal')
    plt.legend()
    plt.grid(True)

    # Plot filtered signal FFT
    plt.subplot(NUMBER_OF_PLOTS, 1, 4)
    filtered_fft = compute_fft(filtered,return_magnitude=True)[1]
    plt.plot(filtered_fft, 'r-', label='FFT of Filtered Signal')
    # plt.stem(original_fft, 'r-', label='FFT of Filtered Signal')
    plt.title('Filtered Signal FFT')
    plt.legend()
    plt.grid(True)

    # Plot kernel
    plt.subplot(NUMBER_OF_PLOTS, 1, 5)
    plt.plot(kernel, 'g-', label='Kernel')
    # plt.stem(kernel, 'g-', label='Kernel')
    plt.title('Convolution Kernel')
    plt.grid(True)

    # # Plot kernel stem
    # plt.grid(True)
    # plt.subplot(7, 1, 6)
    # # plt.stem(kernel, 'g-', label='Kernel')
    # # plt.stem(kernel, 'g-', label='Kernel')
    # plt.title('Convolution Kernel Stem')
    # plt.grid(True)

    # Plot kernel FFT
    plt.subplot(NUMBER_OF_PLOTS, 1, 6)
    kernel_fft = compute_fft(kernel,return_magnitude=True)[1]
    plt.plot(kernel_fft, 'g-', label='FFT of Kernel')
    # plt.stem(kernel_fft, 'g-', label='FFT of Kernel')
    plt.title('Kernel FFT')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_freq_response(filtered_signal_fft, kernel, name1="", axes=None):
    # plot filtered signal FFT and kernel and kernel FFT
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(axes, filtered_signal_fft, 'r-', label=f'Filtered Signal FFT {name1}')
    plt.title('Filtered Signal FFT')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.stem(kernel, 'g-', label='Kernel')
    plt.title('Kernel')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    kernel_fft = compute_fft(kernel,return_magnitude=True)[1]
    plt.plot(kernel_fft, 'g-', label='Kernel FFT')
    plt.title('Kernel FFT')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def signal_sine(n_samples, frequency=1.0, amplitude=1, phase=0):
    """
    Generate a sine wave signal.

    Parameters:
    - n_samples: int, number of samples
    - frequency: float, frequency of the sine wave in Hz
    - amplitude: float, amplitude of the sine wave
    - phase: float, phase shift in radians

    Returns:
    - signal: NumPy array, generated sine wave signal
    """
    t = np.linspace(0, n_samples / 1000, n_samples)  # Time vector
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal

def convolution_tester(n_samples, kernel, m_times=1):
    """Main function to test convolution"""
    # Create test signal
    original_signal = create_test_signal(n_samples)
    # print("original_signal", original_signal)
    # Create a simple averaging kernel
    # kernel = np.ones(kernel_size) / kernel_size
    # print("kernel", kernel)

    # Apply convolution m times
    filtered_signal = apply_convolution(original_signal, kernel, m_times)

    # Plot results
    plot_signals(original_signal, filtered_signal, kernel, m_times)

    return original_signal, filtered_signal, kernel

def sine_hz(n_samples, Hz=1.0, amplitude=1, phase=0, sampling_rate=1.0):
    return signal_sine(n_samples, frequency=Hz*1000/sampling_rate, amplitude=amplitude, phase=phase)
def convolution_tester_noise_int(n_samples, kernel, n, sampling_rate=1.0):

    filtered_signal_fft = np.zeros(n_samples)
    for _ in range(n):
        _signal = np.random.normal(size=n_samples)
        # _signal = sine_hz(n_samples, Hz=2000, amplitude=1, phase=0, sampling_rate=sampling_rate)# + signal_sine(n_samples, frequency=200, amplitude=1, phase=0) + signal_sine(n_samples, frequency=300, amplitude=1, phase=0)
        _signal = np.convolve(_signal, kernel, mode='same')
        fft = compute_fft(_signal, sampling_rate=sampling_rate)[1]
        u = np.zeros(n_samples)
        for i in range(len(fft)):
            u[i] = np.sqrt(fft[i].real**2 + fft[i].imag**2)
        # print(fft, len(fft), n_samples)
        for i in range(len(fft)):
            if u[i] > filtered_signal_fft[i]:
                filtered_signal_fft[i] = u[i]

    plot_freq_response(filtered_signal_fft[:n_samples//2], kernel, name1=f", w_end={sampling_rate/2:.1f}Hz", axes=np.linspace(0, sampling_rate/2, n_samples//2))
    # sampling_rate=20000.0
    # plot_freq_response(sine_hz(n_samples, Hz=1_000, sampling_rate=sampling_rate), kernel, name1=f", t_end={n_samples/sampling_rate:.2f}s")

"""
2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,
 61,67,71,73,79,83,89,97,101,103,107,109,113,127,
 131,137,139,149,151,157,163,167,173,179,181,191,
 193,197,199,211,223,227,229,233,239,241,251,257,
 263,269,271,...
"""

def convolution_frequency_sweep(n_samples, kernel, n_freq=237, sampling_rate=1.0):
    # n_freq is the number of frequencies to sweep
    # Create a frequency vector
    frequencies = np.linspace(0, n_freq, n_samples)

    original_signal = np.zeros(n_samples)
    for f in frequencies:
        _signal = signal_sine(n_samples, frequency=f, amplitude=1, phase=0)
        _signal = apply_convolution(_signal, kernel, m_times=1)
        fft = compute_fft(_signal, sampling_rate=sampling_rate)[1]
        u = np.zeros(n_samples)
        for i in range(len(fft)):
            u[i] = np.sqrt(fft[i].real**2 + fft[i].imag**2)
        for i in range(len(fft)):
            if u[i] > original_signal[i]:
                original_signal[i] = u[i]

    plot_freq_response(original_signal[:n_freq*2], kernel)
    # plot_freq_response(original_signal, kernel)



# dt = 0.01ms => 100kHz sample rate =>(thm. Nyquist) 50kHz max freq
# kernel size -> what's the lowest freq

# Example usage
if __name__ == "__main__":
    # Parameters
    n_samples = 2000  # Number of samples
    # kernel_size = 1000  # Size of the averaging filter
    m_times = 1  # Number of times to apply the filter

    # Define Transfer Function H(z) = b(z) / a(z)
    b = [0.1, 0.2, 0.3]  # Numerator coefficients
    a = [1, -0.5, 0.25]  # Denominator coefficients

    # Define Discrete-Time System
    system = signal.dlti(b, a, dt=1)  # Discrete system

    # Compute Impulse Response
    t, h = signal.dimpulse(system, n=50)  # Get first 50 samples

    # Convert impulse response to a 1D array (FIR coefficients)
    # kernel = np.squeeze(h)

    kernel = np.array([1,0,0,1,0,0,0,0])

    # Run the tester
    # original, filtered, kernel = convolution_tester(
    #     n_samples=n_samples,
    #     kernel=kernel,
    #     m_times=m_times
    # )
    n_noise_res = 1001 #(n_samples//2)+1
    convolution_tester_noise_int(n_samples, kernel, n_noise_res, sampling_rate=20000.0)
    # convolution_frequency_sweep(n_samples, kernel)
