import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

NUMBER_OF_PLOTS = 7

# ----------------------------
# Configurable parameters
# ----------------------------
fs = 10000  # Sampling frequency in Hz
R = 1e4    # Resistance in ohms
C = 1e-7   # Capacitance in farads
N = 201    # Number of FIR filter coefficients (order + 1)

# Calculate the analog RC circuit's frequency response
def H_rc1(f, R, C):
#    s = 1j * 2 * np.pi * f
#    return (s * R * C) / (1 + s * R * C)
    s = 1j * 2 * np.pi * f
    return 1.0 / (1.0 + s * R * C)
def H_rc2(f, R, C):
#    s = 1j * 2 * np.pi * f
#    return (s * R * C) / (1 + s * R * C)
    s = 1j * 2 * np.pi * f
    return (s * R * C) / (1.0 + s * R * C)



# Sample frequencies
frequencies = np.linspace(0, fs / 2, N, endpoint=True)

# Compute half-spectrum for the chosen filter type
H_pos1 = H_rc1(frequencies, R, C)
H_pos2 = H_rc2(frequencies, R, C)

# Build the full frequency response for irfft:
#   - indices 0...(N-1) are the "positive" side
#   - for a real signal, the negative side (N...(2N-2)) is the complex conjugate
#     excluding the 0-th and Nyquist if present.
#
# However, np.fft.irfft only needs length-N data for the half-spectrum,
# so we can insert H_pos directly if it’s already in [0..Nyquist] form.
#
# If H_pos[0] is extremely small, avoid log10(0):
if np.isclose(H_pos1[0], 0.0, atol=1e-20):
    H_pos1[0] = 1e-20
if np.isclose(H_pos2[0], 0.0, atol=1e-20):
    H_pos2[0] = 1e-20

# ----------------------------
# Plot (1): Analog frequency response
# ----------------------------
plt.figure(figsize=(10, 10))

plt.subplot(NUMBER_OF_PLOTS, 1, 1)
plt.plot(frequencies, 20 * np.log10(np.abs(H_pos1)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Analog Filter 1 Frequency Response')
plt.grid()

plt.subplot(NUMBER_OF_PLOTS, 1, 2)
plt.plot(frequencies, 20 * np.log10(np.abs(H_pos2)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Analog Filter 2 Frequency Response')
plt.grid()


# ----------------------------
# 2) Inverse DFT for raw impulse response FIXME: h_raw1 + h_raw2 = [1.0,0,0,0,0,0,0,0,...]
# ----------------------------
# H_pos is the real-FFT representation => use irfft
h_raw1 = np.fft.irfft(H_pos1, n=N)
h_raw2 = np.fft.irfft(H_pos2, n=N)

# Plot (2): Raw impulse response
plt.subplot(NUMBER_OF_PLOTS, 1, 3)
plt.stem(h_raw1, linefmt='C0-', markerfmt='C0o', basefmt=' ')
plt.plot(h_raw1, 'C1')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Raw FIR Impulse Response (no window) 1')
plt.grid()

plt.subplot(NUMBER_OF_PLOTS, 1, 4)
plt.stem(h_raw2, linefmt='C0-', markerfmt='C0o', basefmt=' ')
plt.plot(h_raw2, 'C1')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Raw FIR Impulse Response (no window) 2')
plt.grid()


plt.subplot(NUMBER_OF_PLOTS, 1, 5)
plt.stem(h_raw1 + h_raw2, linefmt='C0-', markerfmt='C0o', basefmt=' ')
plt.plot(h_raw1 + h_raw2, 'C1')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('1+2')
plt.grid()




# ----------------------------
# 3) Apply window and normalize
# ----------------------------
window = np.hanning(N)
h_windowed = h_raw1 * window

# Normalization for unity gain at DC
sum_h = np.sum(h_windowed)
if not np.isclose(sum_h, 0.0, atol=1e-20):
    h_windowed /= sum_h


# Plot (3): Windowed + normalized coefficients
plt.subplot(NUMBER_OF_PLOTS, 1, 6)
plt.stem(h_windowed, linefmt='C0-', markerfmt='C0o', basefmt=' ')
plt.plot(h_windowed, 'C1')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Windowed + Normalized FIR Coefficients')
plt.grid()

# ----------------------------
# 4) Frequency response of the windowed filter
# ----------------------------
w, H_fir = signal.freqz(h_windowed, worN=1024, fs=fs)  # specify fs for direct freq scale

plt.subplot(NUMBER_OF_PLOTS, 1, 7)
plt.plot(w, 20 * np.log10(np.abs(H_fir)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('FIR Filter Frequency Response (after windowing)')
plt.grid()



plt.tight_layout()
plt.show()
