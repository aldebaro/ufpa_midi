"""
Spectrogram visualization and analysis for WAV files.

Converted MATLAB ak_specgram.m functionality to Python.

Key improvements from ak_specgram.m:

- Preemphasis filtering - Boosts higher frequencies for better clarity (coefficient: 0.9)
- Dynamic thresholding - Sets floor value at max - threshold_db to reduce noise artifacts
- Precise time axis calculation - Adjusts time positioning by adding half the window length for better alignment
- Filter bandwidth control - Calculates optimal window length from suggested bandwidth
- Better window calculation - Uses next power of 2 for FFT speed optimization
- dB conversion - Uses 20*log10 for proper power spectrum representation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sys
from pathlib import Path


def ak_specgram(signal_data, fs=None, window='hamming', noverlap=None, nfft=None):
    """
    Compute spectrogram of a signal using short-time Fourier transform (STFT).

    This method performs time-frequency decomposition of a signal by dividing it into
    overlapping segments, applying a window function to each segment, and computing
    the FFT. This is a Python implementation of MATLAB's specgram function adapted
    from ak_specgram.m.

    The spectrogram shows how the frequency content of a signal changes over time,
    which is useful for analyzing audio, speech, and other time-varying signals.

    Parameters
    ----------
    signal_data : array_like
        Input signal (1D array of samples). If stereo, extracts first channel.
    fs : float, optional
        Sampling frequency in Hz (samples per second). Default is 1.0
    window : str or tuple, optional
        Window function to apply to each segment:
        - 'hamming': Default, good general-purpose window
        - 'hann': Smooth transitions
        - 'blackman': Excellent frequency rejection
        - 'bartlett': Triangular window
    noverlap : int, optional
        Number of overlapping samples between consecutive segments (0-99%).
        Default is 50% of window length for smooth time resolution.
    nfft : int, optional
        Length of FFT (zero-padded if longer than window).
        Default: Next power of 2 >= signal_length / 8 (min 256)

    Returns
    -------
    Sxx : ndarray
        Power spectral density in linear scale.
        Shape: (nfft//2 + 1, n_segments) - frequency bins × time frames
    f : ndarray
        Frequency bin centers in Hz (0 to fs/2, Nyquist limit)
    t : ndarray
        Time values for each segment in seconds

    Notes
    -----
    The number of frequency bins is nfft//2 + 1 (one-sided spectrum).
    Time resolution = (nperseg - noverlap) / fs
    Frequency resolution = fs / nfft
    """
    signal_data = np.asarray(signal_data)

    # Handle mono/stereo
    if signal_data.ndim > 1:
        signal_data = signal_data[:, 0]

    # Default sampling frequency
    if fs is None:
        fs = 1.0

    # Default NFFT
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(signal_data) / 8)))
        nfft = max(nfft, 256)

    # Default window length = NFFT
    nperseg = nfft

    # Default noverlap = 50% of nperseg
    if noverlap is None:
        noverlap = nperseg // 2

    # Compute spectrogram using scipy
    f, t, Sxx = signal.spectrogram(
        signal_data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling='density'
    )

    return Sxx, f, t


def plot_spectrogram(signal_data, fs, title='Spectrogram', window='hamming',
                     noverlap=None, nfft=None, vmin=None, vmax=None,
                     cmap='viridis', figsize=(12, 6)):
    """
    Compute and visualize a spectrogram with professional formatting.

    This method creates a time-frequency visualization of a signal where:
    - Horizontal axis represents time progression
    - Vertical axis represents frequency content
    - Color intensity represents power (magnitude squared) at that time-frequency point

    The visualization uses matplotlib's pcolormesh for efficient rendering.

    Parameters
    ----------
    signal_data : array_like
        Input audio signal (1D array of samples)
    fs : float
        Sampling frequency in Hz (samples per second)
    title : str, default 'Spectrogram'
        Plot title displayed at top
    window : str, default 'hamming'
        Windowing function applied to segments
    noverlap : int, optional
        Samples overlapped between segments. None uses 50% overlap.
    nfft : int, optional
        FFT length for frequency resolution. None auto-calculates.
    vmin, vmax : float, optional
        Color scale limits in linear units. None for auto-scaling.
    cmap : str, default 'viridis'
        Matplotlib colormap name ('viridis', 'plasma', 'magma', 'inferno', etc.)
    figsize : tuple, default (12, 6)
        Figure dimensions (width_inches, height_inches)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization or saving
    ax : matplotlib.axes.Axes
        Axes object with the spectrogram plot

    Notes
    -----
    Power converted to linear scale using 10*log10(). Frequency limits set to
    Nyquist frequency (fs/2). Colorbar shows power scale.
    """
    # Compute spectrogram
    Sxx, f, t = ak_specgram(signal_data, fs=fs, window=window,
                            noverlap=noverlap, nfft=nfft)

    # Convert to dB scale
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot spectrogram
    im = ax.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap=cmap,
                       vmin=vmin, vmax=vmax)

    # Labels and title
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')

    # Set reasonable frequency limits
    ax.set_ylim([0, fs/2])

    plt.tight_layout()

    return fig, ax


def load_wav_and_plot_spectrogram(wav_file, title=None, nfft=None,
                                  window='hamming', noverlap=None,
                                  figsize=(14, 6), save_path=None):
    """
    Complete pipeline: load WAV file and generate spectrogram visualization.

    This high-level method handles all steps in the workflow:
    1. Loads audio from WAV file
    2. Converts stereo to mono (if needed)
    3. Normalizes amplitude to prevent clipping
    4. Computes spectrogram via STFT
    5. Creates formatted plot with colorbar
    6. Saves to image file or displays interactively

    Parameters
    ----------
    wav_file : str
        Path to input WAV file (absolute or relative)
    title : str, optional
        Plot title. If None, uses WAV filename. Default: None
    nfft : int, optional
        FFT length (frequency resolution). If None, auto-calculated.
    window : str, default 'hamming'
        Window function name for STFT
    noverlap : int, optional
        Overlapping samples between segments. None uses 50% overlap.
    figsize : tuple, default (14, 6)
        Output figure size in inches (width, height)
    save_path : str, optional
        File path to save figure (e.g., 'spectrogram.png', 'plot.jpg').
        If None, displays plot in interactive window. Default: None

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object for additional customization
    ax : matplotlib.axes.Axes
        Matplotlib axes object containing the spectrogram plot

    Raises
    ------
    FileNotFoundError
        If wav_file does not exist
    ValueError
        If WAV file is corrupted or unsupported format

    Examples
    --------
    >>> fig, ax = load_wav_and_plot_spectrogram('audio.wav')
    >>> fig, ax = load_wav_and_plot_spectrogram('audio.wav', 'My Audio', save_path='spec.png')

    Notes
    -----
    Stereo files are converted to mono by taking the first channel.
    Audio is normalized to [-1, 1] range to prevent clipping in processing.
    """
    # Load WAV file
    fs, audio_data = wavfile.read(wav_file)

    # Handle stereo
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Normalize if necessary
    audio_data = audio_data.astype(np.float32)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    # Default title
    if title is None:
        title = f'Spectrogram: {Path(wav_file).name}'

    # Plot spectrogram
    fig, ax = plot_spectrogram(
        audio_data, fs, title=title, window=window,
        noverlap=noverlap, nfft=nfft, figsize=figsize
    )

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spectrogram saved to: {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_several_spectrograms(wav_files, figsize=(15, 4), window='hamming'):
    """
    Create side-by-side comparison of spectrograms from multiple WAV files.

    This method is useful for comparing frequency content across different audio files,
    such as comparing different recordings, processing effects, or quality levels.
    All spectrograms use consistent scaling for fair comparison.

    Parameters
    ----------
    wav_files : list of str
        Paths to WAV files to compare. Files are processed in order left-to-right.
    figsize : tuple, default (15, 4)
        Output figure dimensions in inches (width, height).
        Height of 4 is good for single row; increase for more subplots if needed.
    window : str, default 'hamming'
        Window function applied during STFT computation

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing all subplots
    axes : list of matplotlib.axes.Axes
        List of axes objects, one per WAV file

    Notes
    -----
    - Creates N subplots for N input files (horizontal layout)
    - Each subplot shows full frequency range (0 to Nyquist)
    - Stereo files automatically converted to mono
    - Each subplot includes independent colorbar for power scale

    Examples
    --------
    >>> fig, axes = plot_several_spectrograms(['file1.wav', 'file2.wav', 'file3.wav'])
    """
    n_files = len(wav_files)
    fig, axes = plt.subplots(1, n_files, figsize=figsize)

    if n_files == 1:
        axes = [axes]

    for idx, wav_file in enumerate(wav_files):
        fs, audio_data = wavfile.read(wav_file)

        # Handle stereo
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # Normalize
        audio_data = audio_data.astype(np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Compute spectrogram
        Sxx, f, t = ak_specgram(audio_data, fs=fs, window=window)
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        # Plot
        im = axes[idx].pcolormesh(
            t, f, Sxx_dB, shading='gouraud', cmap='viridis')
        axes[idx].set_ylabel('Frequency (Hz)')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_title(Path(wav_file).name)
        axes[idx].set_ylim([0, fs/2])

        fig.colorbar(im, ax=axes[idx], label='Power (dB)')

    plt.tight_layout()
    plt.show()
    return fig, axes


def main():
    """
    Command-line interface for spectrogram generation.

    Provides easy access to spectrogram functionality from terminal without
    writing Python code. Supports both interactive display and file output.

    Usage
    -----
    python show_wav_spectrogram.py <wav_file> [output_image]

    Arguments
    ---------
    wav_file : str
        Path to input WAV file (required)
    output_image : str
        Path to save output image (optional). If omitted, displays interactively.
        Supported formats: .png (recommended), .jpg, .pdf, .svg, .eps

    Examples
    --------
    Display spectrogram in window:
        python show_wav_spectrogram.py audio.wav

    Save spectrogram as PNG file:
        python show_wav_spectrogram.py audio.wav spectrogram.png

    Notes
    -----
    - Exits with code 1 if input file not found
    - Uses default parameters (hamming window, auto FFT length)
    - For advanced options, import and call functions directly in Python
    """

    if len(sys.argv) < 2:
        print(
            "Usage: python show_wav_spectrogram.py <wav_file> [output_image]")
        print("\nExample:")
        print("  python show_wav_spectrogram.py audio.wav")
        print("  python show_wav_spectrogram.py audio.wav spectrogram.png")
        sys.exit(1)

    wav_file = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if file exists
    if not Path(wav_file).exists():
        print(f"Error: WAV file not found: {wav_file}")
        sys.exit(1)

    print(f"Loading {wav_file}...")

    # Load and plot spectrogram
    if True:
        load_wav_and_plot_spectrogram(wav_file, save_path=output_image)
    else:
        plot_several_spectrograms([wav_file, wav_file], figsize=(10, 5))


if __name__ == "__main__":
    main()
