"""
Creates a WAV file from a MIDI file.
Uses fluidsynth for high-quality audio synthesis.
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
import subprocess
from pathlib import Path


def convert_midi_to_wav_fluidsynth(midi_file, output_wav, soundfont=None, sample_rate=44100):
    """
    Convert MIDI to WAV using fluidsynth (requires fluidsynth CLI installed).

    Args:
        midi_file: Path to input MIDI file
        output_wav: Path to output WAV file
        soundfont: Path to soundfont file (optional)
        sample_rate: Sample rate for output WAV (default 44100)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build fluidsynth command
        cmd = ['fluidsynth', '-ni']

        # Add soundfont if specified
        if soundfont and os.path.exists(soundfont):
            cmd.extend([soundfont])

        # Add MIDI file and output file
        cmd.extend(['-F', output_wav, midi_file])

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully converted {midi_file} to {output_wav}")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: fluidsynth not found. Please install fluidsynth.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_midi_to_wav_pydub(midi_file, output_wav):
    """
    Convert MIDI to WAV using pydub and simpleaudio.

    Args:
        midi_file: Path to input MIDI file
        output_wav: Path to output WAV file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from pydub import AudioSegment

        # Note: pydub doesn't directly support MIDI
        # This is a placeholder for informational purposes
        print("Note: pydub doesn't have native MIDI support.")
        print("Consider using fluidsynth or installing additional dependencies.")
        return False
    except ImportError:
        return False


def convert_midi_to_wav_pretty_midi(midi_file, output_wav, sample_rate=44100):
    """
    Convert MIDI to WAV using pretty_midi and simpleaudio.

    Args:
        midi_file: Path to input MIDI file
        output_wav: Path to output WAV file
        sample_rate: Sample rate for output WAV (default 44100)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pretty_midi

        # Load MIDI file
        midi = pretty_midi.PrettyMIDI(midi_file)

        # Synthesize to audio
        print(f"Synthesizing {midi_file}...")
        audio = midi.synthesize(fs=sample_rate)

        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        # Convert to 16-bit PCM
        audio_int16 = np.int16(audio * 32767)

        # Write to WAV file
        wavfile.write(output_wav, sample_rate, audio_int16)

        print(f"Successfully converted {midi_file} to {output_wav}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
        return True

    except ImportError:
        print("Error: pretty_midi not installed.")
        print("Install with: pip install pretty_midi")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function to convert MIDI to WAV."""

    if len(sys.argv) < 2:
        print(
            "Usage: python midi_to_wav.py <midi_file> [output_wav] [--method {fluidsynth|pretty_midi}]")
        print("\nExamples:")
        print("  python midi_to_wav.py input.mid")
        print("  python midi_to_wav.py input.mid output.wav")
        print("  python midi_to_wav.py input.mid output.wav --method pretty_midi")
        print("  python midi_to_wav.py input.mid output.wav --method fluidsynth")
        sys.exit(1)

    midi_file = sys.argv[1]

    # Check if MIDI file exists
    if not os.path.exists(midi_file):
        print(f"Error: MIDI file not found: {midi_file}")
        sys.exit(1)

    # Generate output filename if not provided
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_wav = sys.argv[2]
    else:
        output_wav = os.path.splitext(midi_file)[0] + '.wav'

    # Determine conversion method
    method = 'pretty_midi'
    for i, arg in enumerate(sys.argv):
        if arg == '--method' and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]

    print(f"Converting {midi_file} to {output_wav} using {method}...")

    if method == 'fluidsynth':
        success = convert_midi_to_wav_fluidsynth(midi_file, output_wav)
    elif method == 'pretty_midi':
        success = convert_midi_to_wav_pretty_midi(midi_file, output_wav)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)

    if success:
        print(f"\n✓ WAV file ready for Audacity: {output_wav}")
        sys.exit(0)
    else:
        print(f"\n✗ Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
