"""
Data Augmentation Module for BirdCLEF Audio Processing
=====================================================

Includes various audio augmentation techniques for bird sound classification:
- Mixup: Linear interpolation between two audio samples
- Background noise: Add gaussian, pink, brown noise
- Pitch/Time shifting: Change pitch and temporal characteristics
- Frequency/Time masking: SpecAugment-style masking
- Color jitter: Spectral modifications
- Random clip selection: Extract 5-second segments

Usage:
    from data_augmentation import AudioAugmenter
    
    augmenter = AudioAugmenter(
        mixup=True,
        background_noise=True,
        pitch_shift=True,
        time_shift=True,
        freq_masking=True,
        time_masking=True,
        color_jitter=True,
        random_clip=True
    )
    
    augmented_data = augmenter.process_directory("birdclef-2023/train_audio")
"""

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os
from pathlib import Path
import random
import json
import shutil
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class NoiseGenerator:
    """Generate different types of noise for augmentation"""
    
    @staticmethod
    def gaussian_noise(duration: float, sr: int, amplitude: float = 0.01) -> np.ndarray:
        """Generate white/gaussian noise"""
        samples = int(duration * sr)
        return np.random.normal(0, amplitude, samples).astype(np.float32)
    
    @staticmethod
    def pink_noise(duration: float, sr: int, amplitude: float = 0.01) -> np.ndarray:
        """Generate pink noise (1/f noise)"""
        samples = int(duration * sr)
        # Generate white noise
        white = np.random.randn(samples)
        
        # Apply pink noise filter (approximate)
        # Pink noise has power spectral density proportional to 1/f
        freqs = np.fft.fftfreq(samples, 1/sr)
        freqs[0] = 1e-6  # Avoid division by zero
        
        # Create 1/f filter
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_filter[0] = 1
        
        # Apply filter
        white_fft = np.fft.fft(white)
        pink_fft = white_fft * pink_filter
        pink = np.real(np.fft.ifft(pink_fft))
        
        # Normalize and scale
        pink = pink / np.std(pink) * amplitude
        return pink.astype(np.float32)
    
    @staticmethod
    def brown_noise(duration: float, sr: int, amplitude: float = 0.01) -> np.ndarray:
        """Generate brown noise (1/f^2 noise)"""
        samples = int(duration * sr)
        white = np.random.randn(samples)
        
        freqs = np.fft.fftfreq(samples, 1/sr)
        freqs[0] = 1e-6
        
        # Create 1/f^2 filter
        brown_filter = 1 / np.abs(freqs)
        brown_filter[0] = 1
        
        white_fft = np.fft.fft(white)
        brown_fft = white_fft * brown_filter
        brown = np.real(np.fft.ifft(brown_fft))
        
        brown = brown / np.std(brown) * amplitude
        return brown.astype(np.float32)


class AudioAugmenter:
    """Main class for audio data augmentation"""
    
    def __init__(self, 
                 mixup: bool = False,
                 background_noise: bool = False,
                 pitch_shift: bool = False,
                 time_shift: bool = False,
                 freq_masking: bool = False,
                 time_masking: bool = False,
                 color_jitter: bool = False,
                 random_clip: bool = False,
                 output_dir: str = "birdclef-2023/augmented",
                 target_sr: int = 32000,
                 clip_duration: float = 5.0):
        """
        Initialize AudioAugmenter with augmentation flags
        
        Args:
            mixup: Apply mixup augmentation
            background_noise: Add background noise
            pitch_shift: Apply pitch shifting
            time_shift: Apply time shifting
            freq_masking: Apply frequency masking
            time_masking: Apply time masking
            color_jitter: Apply spectral color jitter
            random_clip: Extract random clips
            output_dir: Directory to save augmented data
            target_sr: Target sample rate
            clip_duration: Duration of clips in seconds
        """
        self.mixup = mixup
        self.background_noise = background_noise
        self.pitch_shift = pitch_shift
        self.time_shift = time_shift
        self.freq_masking = freq_masking
        self.time_masking = time_masking
        self.color_jitter = color_jitter
        self.random_clip = random_clip
        
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        self.clip_duration = clip_duration
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_audio").mkdir(exist_ok=True)
        
        # Initialize noise generator
        self.noise_gen = NoiseGenerator()
        
        # Augmentation parameters
        self.aug_params = {
            'mixup_alpha': 0.4,  # Beta distribution parameter
            'noise_amplitude': (0.005, 0.02),  # Noise amplitude range
            'pitch_shift_range': (-2, 2),  # Semitones
            'time_stretch_range': (0.8, 1.2),  # Speed factor
            'freq_mask_param': 27,  # Frequency masking parameter
            'time_mask_param': 100,  # Time masking parameter
            'n_freq_masks': 2,  # Number of frequency masks
            'n_time_masks': 2,  # Number of time masks
            'spectral_jitter_strength': 0.1  # Color jitter strength
        }
        
        self.metadata_records = []
    
    def apply_mixup(self, audio1: np.ndarray, audio2: np.ndarray, label1: str, label2: str) -> Tuple[np.ndarray, str, Dict]:
        """
        Apply mixup augmentation between two audio samples
        
        Args:
            audio1, audio2: Audio arrays
            label1, label2: Corresponding labels
            
        Returns:
            Mixed audio, combined label, metadata
        """
        # Sample mixing coefficient from Beta distribution
        alpha = self.aug_params['mixup_alpha']
        lam = np.random.beta(alpha, alpha)
        
        # Ensure same length by padding shorter audio
        max_len = max(len(audio1), len(audio2))
        if len(audio1) < max_len:
            audio1 = np.pad(audio1, (0, max_len - len(audio1)), 'constant')
        if len(audio2) < max_len:
            audio2 = np.pad(audio2, (0, max_len - len(audio2)), 'constant')
        
        # Mix audio
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        
        # Create combined label
        combined_label = f"mixup_{label1}_{label2}_{lam:.3f}"
        
        metadata = {
            'augmentation': 'mixup',
            'source_labels': [label1, label2],
            'mixing_coefficient': float(lam),
            'primary_weight': float(lam),
            'secondary_weight': float(1 - lam)
        }
        
        return mixed_audio, combined_label, metadata
    
    def add_background_noise(self, audio: np.ndarray, noise_type: str = 'gaussian') -> Tuple[np.ndarray, Dict]:
        """
        Add background noise to audio
        
        Args:
            audio: Input audio array
            noise_type: Type of noise ('gaussian', 'pink', 'brown')
            
        Returns:
            Noisy audio, metadata
        """
        duration = len(audio) / self.target_sr
        amplitude = np.random.uniform(*self.aug_params['noise_amplitude'])
        
        if noise_type == 'gaussian':
            noise = self.noise_gen.gaussian_noise(duration, self.target_sr, amplitude)
        elif noise_type == 'pink':
            noise = self.noise_gen.pink_noise(duration, self.target_sr, amplitude)
        elif noise_type == 'brown':
            noise = self.noise_gen.brown_noise(duration, self.target_sr, amplitude)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Ensure same length
        min_len = min(len(audio), len(noise))
        noisy_audio = audio[:min_len] + noise[:min_len]
        
        metadata = {
            'augmentation': 'background_noise',
            'noise_type': noise_type,
            'noise_amplitude': float(amplitude)
        }
        
        return noisy_audio, metadata
    
    def apply_pitch_shift(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply pitch shifting"""
        n_steps = np.random.uniform(*self.aug_params['pitch_shift_range'])
        shifted_audio = librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=n_steps)
        
        metadata = {
            'augmentation': 'pitch_shift',
            'semitones': float(n_steps)
        }
        
        return shifted_audio, metadata
    
    def apply_time_shift(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply time stretching/compression"""
        rate = np.random.uniform(*self.aug_params['time_stretch_range'])
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
        
        metadata = {
            'augmentation': 'time_shift',
            'stretch_rate': float(rate)
        }
        
        return stretched_audio, metadata
    
    def apply_spec_augment(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply SpecAugment (frequency and time masking)"""
        # Convert to spectrogram
        S = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(S)
        phase = np.angle(S)
        
        # Apply frequency masking
        if self.freq_masking:
            for _ in range(self.aug_params['n_freq_masks']):
                f = np.random.uniform(0, self.aug_params['freq_mask_param'])
                f = int(f)
                f0 = np.random.randint(0, magnitude.shape[0] - f)
                magnitude[f0:f0+f, :] = 0
        
        # Apply time masking
        if self.time_masking:
            for _ in range(self.aug_params['n_time_masks']):
                t = np.random.uniform(0, self.aug_params['time_mask_param'])
                t = int(t)
                if magnitude.shape[1] > t:
                    t0 = np.random.randint(0, magnitude.shape[1] - t)
                    magnitude[:, t0:t0+t] = 0
        
        # Convert back to audio
        masked_S = magnitude * np.exp(1j * phase)
        masked_audio = librosa.istft(masked_S, hop_length=512)
        
        metadata = {
            'augmentation': 'spec_augment',
            'freq_masking': self.freq_masking,
            'time_masking': self.time_masking,
            'n_freq_masks': self.aug_params['n_freq_masks'] if self.freq_masking else 0,
            'n_time_masks': self.aug_params['n_time_masks'] if self.time_masking else 0
        }
        
        return masked_audio, metadata
    
    def apply_color_jitter(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply spectral color jitter"""
        # Convert to spectrogram
        S = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(S)
        phase = np.angle(S)
        
        # Apply random spectral modification
        strength = self.aug_params['spectral_jitter_strength']
        freq_bins = magnitude.shape[0]
        
        # Create random frequency response
        jitter_response = 1 + np.random.uniform(-strength, strength, freq_bins)
        jitter_response = np.maximum(jitter_response, 0.1)  # Avoid complete nulling
        
        # Apply to magnitude spectrum
        jittered_magnitude = magnitude * jitter_response.reshape(-1, 1)
        
        # Convert back to audio
        jittered_S = jittered_magnitude * np.exp(1j * phase)
        jittered_audio = librosa.istft(jittered_S, hop_length=512)
        
        metadata = {
            'augmentation': 'color_jitter',
            'jitter_strength': float(strength),
            'freq_response_range': [float(jitter_response.min()), float(jitter_response.max())]
        }
        
        return jittered_audio, metadata
    
    def extract_random_clip(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Extract random clip of specified duration"""
        audio_duration = len(audio) / self.target_sr
        
        if audio_duration <= self.clip_duration:
            # Audio is shorter than clip duration, pad with zeros
            target_samples = int(self.clip_duration * self.target_sr)
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)), 'constant')
            clip = audio[:target_samples]
            start_time = 0.0
        else:
            # Extract random clip
            max_start = audio_duration - self.clip_duration
            start_time = np.random.uniform(0, max_start)
            start_sample = int(start_time * self.target_sr)
            end_sample = start_sample + int(self.clip_duration * self.target_sr)
            clip = audio[start_sample:end_sample]
        
        metadata = {
            'augmentation': 'random_clip',
            'clip_duration': float(self.clip_duration),
            'start_time': float(start_time),
            'original_duration': float(audio_duration)
        }
        
        return clip, metadata
    
    def process_audio_file(self, file_path: Path, primary_label: str, original_metadata: Dict = None) -> List[Dict]:
        """Process a single audio file with all enabled augmentations"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            original_filename = file_path.stem
            
            # Set default metadata if not provided
            if original_metadata is None:
                original_metadata = {
                    'latitude': '',
                    'longitude': '',
                    'scientific_name': '',
                    'common_name': '',
                    'author': '',
                    'license': '',
                    'rating': '',
                    'url': ''
                }
            
            results = []
            
            # Apply random clip first if enabled
            if self.random_clip:
                audio, clip_metadata = self.extract_random_clip(audio)
            else:
                clip_metadata = None
            
            # Apply individual augmentations
            augmentations = []
            
            if self.background_noise:
                noise_types = ['gaussian', 'pink', 'brown']
                for noise_type in noise_types:
                    noisy_audio, noise_meta = self.add_background_noise(audio, noise_type)
                    augmentations.append(('noise_' + noise_type, noisy_audio, noise_meta))
            
            if self.pitch_shift:
                pitched_audio, pitch_meta = self.apply_pitch_shift(audio)
                augmentations.append(('pitch_shift', pitched_audio, pitch_meta))
            
            if self.time_shift:
                stretched_audio, stretch_meta = self.apply_time_shift(audio)
                augmentations.append(('time_shift', stretched_audio, stretch_meta))
            
            if self.freq_masking or self.time_masking:
                masked_audio, mask_meta = self.apply_spec_augment(audio)
                augmentations.append(('spec_augment', masked_audio, mask_meta))
            
            if self.color_jitter:
                jittered_audio, jitter_meta = self.apply_color_jitter(audio)
                augmentations.append(('color_jitter', jittered_audio, jitter_meta))
            
            # Save augmented files
            for aug_name, aug_audio, aug_meta in augmentations:
                # Create filename
                aug_filename = f"{original_filename}_{aug_name}.ogg"
                
                # Create species directory
                species_dir = self.output_dir / "train_audio" / primary_label
                species_dir.mkdir(exist_ok=True)
                
                # Save audio file
                output_path = species_dir / aug_filename
                sf.write(output_path, aug_audio, self.target_sr, format='OGG', subtype='VORBIS')
                
                # Create metadata record with inherited metadata
                duration = len(aug_audio) / sr
                metadata_record = {
                    'primary_label': primary_label,
                    'secondary_labels': '[]',
                    'type': "['augmented']",
                    'latitude': original_metadata.get('latitude', ''),
                    'longitude': original_metadata.get('longitude', ''),
                    'scientific_name': original_metadata.get('scientific_name', ''),
                    'common_name': original_metadata.get('common_name', ''),
                    'author': f"AudioAugmenter (orig: {original_metadata.get('author', 'Unknown')})",
                    'license': original_metadata.get('license', 'Generated'),
                    'rating': 'synthetic',
                    'url': original_metadata.get('url', ''),
                    'filename': f"{primary_label}/{aug_filename}",
                    'duration': duration,
                    'original_file': str(file_path),
                    'augmentation_metadata': {
                        'clip_extraction': clip_metadata,
                        'augmentation': aug_meta
                    }
                }
                
                results.append(metadata_record)
            
            return results
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    def process_mixup_pairs(self, audio_files: List[Path], labels: List[str], original_metadata_dict: Dict = None, n_mixup: int = 100) -> List[Dict]:
        """Generate mixup samples between different species"""
        if not self.mixup or len(audio_files) < 2:
            return []
        
        if original_metadata_dict is None:
            original_metadata_dict = {}
        
        results = []
        
        for _ in range(n_mixup):
            # Randomly select two different files
            idx1, idx2 = np.random.choice(len(audio_files), 2, replace=False)
            file1, file2 = audio_files[idx1], audio_files[idx2]
            label1, label2 = labels[idx1], labels[idx2]
            
            try:
                # Load audio files
                audio1, _ = librosa.load(file1, sr=self.target_sr)
                audio2, _ = librosa.load(file2, sr=self.target_sr)
                
                # Apply random clipping if enabled
                if self.random_clip:
                    audio1, _ = self.extract_random_clip(audio1)
                    audio2, _ = self.extract_random_clip(audio2)
                
                # Apply mixup
                mixed_audio, mixed_label, mix_meta = self.apply_mixup(audio1, audio2, label1, label2)
                
                # Save mixed audio
                mixup_dir = self.output_dir / "train_audio" / "mixup"
                mixup_dir.mkdir(exist_ok=True)
                
                mixed_filename = f"{file1.stem}_{file2.stem}_mixup_{mix_meta['mixing_coefficient']:.3f}.ogg"
                output_path = mixup_dir / mixed_filename
                
                sf.write(output_path, mixed_audio, self.target_sr, format='OGG', subtype='VORBIS')
                
                # Create metadata record - inherit from primary file (file1)
                duration = len(mixed_audio) / self.target_sr
                primary_metadata = original_metadata_dict.get(file1.name, {})
                secondary_metadata = original_metadata_dict.get(file2.name, {})
                
                metadata_record = {
                    'primary_label': label1,
                    'secondary_labels': f"['{label2}']",
                    'type': "['mixup']",
                    'latitude': primary_metadata.get('latitude', ''),
                    'longitude': primary_metadata.get('longitude', ''),
                    'scientific_name': primary_metadata.get('scientific_name', ''),
                    'common_name': primary_metadata.get('common_name', ''),
                    'author': f"AudioAugmenter (mixed: {primary_metadata.get('author', 'Unknown')} + {secondary_metadata.get('author', 'Unknown')})",
                    'license': primary_metadata.get('license', 'Generated'),
                    'rating': 'synthetic',
                    'url': primary_metadata.get('url', ''),
                    'filename': f"mixup/{mixed_filename}",
                    'duration': duration,
                    'original_files': [str(file1), str(file2)],
                    'augmentation_metadata': mix_meta
                }
                
                results.append(metadata_record)
                
            except Exception as e:
                print(f"Error creating mixup from {file1} and {file2}: {e}")
                continue
        
        return results
    
    def process_directory(self, input_dir: str, metadata_csv: str = None, species_filter: List[str] = None) -> str:
        """
        Process all audio files in directory with augmentations
        
        Args:
            input_dir: Directory containing audio files organized by species
            metadata_csv: Optional path to existing metadata CSV
            species_filter: Optional list of species directories to process (e.g., ['barswa', 'comsan'])
                           If None, processes all species directories
            
        Returns:
            Path to generated metadata CSV
        """
        input_path = Path(input_dir)
        all_metadata = []
        all_audio_files = []
        all_labels = []
        
        # Load original metadata if provided
        original_metadata_dict = {}
        if metadata_csv and Path(metadata_csv).exists():
            print(f"📊 Loading original metadata from {metadata_csv}")
            orig_df = pd.read_csv(metadata_csv)
            # Create lookup dictionary: filename -> metadata
            for _, row in orig_df.iterrows():
                filename = Path(row['filename']).name  # Just the filename without path
                original_metadata_dict[filename] = {
                    'latitude': row.get('latitude', ''),
                    'longitude': row.get('longitude', ''),
                    'scientific_name': row.get('scientific_name', ''),
                    'common_name': row.get('common_name', ''),
                    'author': row.get('author', ''),
                    'license': row.get('license', ''),
                    'rating': row.get('rating', ''),
                    'url': row.get('url', '')
                }
            print(f"📊 Loaded metadata for {len(original_metadata_dict)} original files")
        
        print(f"Processing audio files from {input_dir}")
        if species_filter:
            print(f"Species filter: {species_filter}")
        enabled_augs = [name for name, enabled in [
            ('mixup', self.mixup),
            ('background_noise', self.background_noise), 
            ('pitch_shift', self.pitch_shift),
            ('time_shift', self.time_shift),
            ('freq_masking', self.freq_masking),
            ('time_masking', self.time_masking),
            ('color_jitter', self.color_jitter),
            ('random_clip', self.random_clip)
        ] if enabled]
        print(f"Enabled augmentations: {enabled_augs}")
        
        # Process each species directory
        for species_dir in input_path.iterdir():
            if not species_dir.is_dir():
                continue
                
            primary_label = species_dir.name
            
            # Skip if species filter is provided and this species is not in the filter
            if species_filter and primary_label not in species_filter:
                continue
            print(f"\n📁 Processing {primary_label}...")
            
            # Get all audio files for this species
            audio_files = list(species_dir.glob("*.ogg")) + list(species_dir.glob("*.mp3")) + list(species_dir.glob("*.wav"))
            
            for audio_file in audio_files:
                # Get original metadata for this file
                file_key = audio_file.name
                orig_metadata = original_metadata_dict.get(file_key, None)
                
                # Process individual file augmentations
                file_results = self.process_audio_file(audio_file, primary_label, orig_metadata)
                all_metadata.extend(file_results)
                
                # Collect for mixup processing
                all_audio_files.append(audio_file)
                all_labels.append(primary_label)
            
            print(f"Processed {len(audio_files)} files for {primary_label}")
        
        # Process mixup pairs
        if self.mixup and len(all_audio_files) > 1:
            print(f"\n🎛️ Generating mixup samples...")
            mixup_results = self.process_mixup_pairs(all_audio_files, all_labels, original_metadata_dict, n_mixup=100)
            all_metadata.extend(mixup_results)
            print(f"Generated {len(mixup_results)} mixup samples")
        
        # Save metadata
        if all_metadata:
            df = pd.DataFrame(all_metadata)
            
            # Ensure BirdCLEF column order with duration
            birdclef_columns = ['primary_label', 'secondary_labels', 'type', 'latitude', 'longitude', 
                              'scientific_name', 'common_name', 'author', 'license', 'rating', 'url', 'filename', 'duration']
            
            for col in birdclef_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Reorder columns to match BirdCLEF format exactly
            output_df = df[birdclef_columns].copy()
            
            # Save to CSV
            metadata_path = self.output_dir / "augmented_metadata.csv"
            output_df.to_csv(metadata_path, index=False)
            
            print(f"\nGenerated {len(all_metadata)} augmented samples")
            print(f"Metadata saved to: {metadata_path}")
            
            # Save augmentation config
            config_path = self.output_dir / "augmentation_config.json"
            config = {
                'enabled_augmentations': {
                    'mixup': self.mixup,
                    'background_noise': self.background_noise,
                    'pitch_shift': self.pitch_shift,
                    'time_shift': self.time_shift,
                    'freq_masking': self.freq_masking,
                    'time_masking': self.time_masking,
                    'color_jitter': self.color_jitter,
                    'random_clip': self.random_clip
                },
                'parameters': self.aug_params,
                'target_sr': self.target_sr,
                'clip_duration': self.clip_duration,
                'total_samples': len(all_metadata)
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return str(metadata_path)
        
        else:
            print("No augmented samples generated")
            return None

    def create_filtered_dataset(self, 
                              base_metadata: str,
                              filter_config: Dict,
                              output_name: str = "filtered") -> str:
        """
        Create a filtered dataset combining base and augmented data
        """
        
        # Get augmented metadata if it exists
        aug_metadata = self.output_dir / "augmented_metadata.csv"
        aug_metadata_str = str(aug_metadata) if aug_metadata.exists() else None
        
        # Create output paths
        output_metadata = f"birdclef-2023/{output_name}_metadata.csv"
        output_audio_dir = f"birdclef-2023/{output_name}_audio"
        
        # Filter and combine data
        result_path = filter_and_concatenate_data(
            base_metadata_path=base_metadata,
            augmented_metadata_path=aug_metadata_str,
            output_metadata_path=output_metadata,
            filter_conditions=filter_config,
            target_audio_dir=output_audio_dir
        )
        
        return result_path


def filter_and_concatenate_data(base_metadata_path: str,
                               augmented_metadata_path: str = None,
                               output_metadata_path: str = "birdclef-2023/combined_metadata.csv",
                               filter_conditions: Dict = None,
                               copy_audio_files: bool = True,
                               target_audio_dir: str = "birdclef-2023/combined_audio") -> str:
    """
    Filter and concatenate audio data from multiple sources
    
    Args:
        base_metadata_path: Path to base/original metadata CSV
        augmented_metadata_path: Path to augmented metadata CSV (optional)
        output_metadata_path: Where to save combined metadata
        filter_conditions: Dictionary of filter conditions
        copy_audio_files: Whether to copy/organize audio files
        target_audio_dir: Where to organize audio files
        
    Returns:
        Path to combined metadata CSV
        
    Filter conditions examples:
    {
        'species': ['barswa', 'comsan', 'eaywag1'],  # Only these species
        'min_duration': 5.0,  # Minimum duration in seconds
        'max_duration': 60.0,  # Maximum duration in seconds
        'rating': ['A', 'B'],  # Only high quality ratings
        'exclude_augmented': True,  # Exclude augmented samples
        'augmentation_types': ['mixup', 'noise_gaussian'],  # Only these augmentations
        'author': ['AudioAugmenter'],  # Specific authors
        'min_samples_per_species': 10,  # Minimum samples per species
        'max_samples_per_species': 100,  # Maximum samples per species
        'random_sample': 0.5,  # Random sample fraction
    }
    """
    
    # Load base metadata
    base_df = pd.read_csv(base_metadata_path)
    print(f"📊 Loaded {len(base_df)} records from base metadata")
    
    # Initialize combined dataframe with base data
    combined_df = base_df.copy()
    
    # Add augmented data if provided
    if augmented_metadata_path and Path(augmented_metadata_path).exists():
        aug_df = pd.read_csv(augmented_metadata_path)
        print(f"📊 Loaded {len(aug_df)} augmented records")
        combined_df = pd.concat([combined_df, aug_df], ignore_index=True)
    
    # Apply filters if provided
    if filter_conditions:
        original_len = len(combined_df)
        combined_df = apply_filters(combined_df, filter_conditions)
        print(f"🔍 Filtered from {original_len} to {len(combined_df)} records")
    
    # Copy/organize audio files if requested
    if copy_audio_files:
        organize_audio_files(combined_df, target_audio_dir)
    
    # Save combined metadata
    Path(output_metadata_path).parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_metadata_path, index=False)
    print(f"💾 Saved combined metadata to {output_metadata_path}")
    
    return output_metadata_path


def apply_filters(df: pd.DataFrame, conditions: Dict) -> pd.DataFrame:
    """Apply filtering conditions to dataframe"""
    filtered_df = df.copy()
    
    # Filter by species
    if 'species' in conditions:
        filtered_df = filtered_df[filtered_df['primary_label'].isin(conditions['species'])]
        print(f"  🐦 Species filter: kept {len(filtered_df)} records")
    
    # Filter by duration
    if 'min_duration' in conditions:
        filtered_df = filtered_df[filtered_df['duration'] >= conditions['min_duration']]
        print(f"  ⏱️ Min duration filter: kept {len(filtered_df)} records")
    
    if 'max_duration' in conditions:
        filtered_df = filtered_df[filtered_df['duration'] <= conditions['max_duration']]
        print(f"  ⏱️ Max duration filter: kept {len(filtered_df)} records")
    
    # Filter by rating
    if 'rating' in conditions:
        # Handle both string ratings and 'synthetic' for augmented data
        rating_values = conditions['rating']
        if isinstance(rating_values, list):
            filtered_df = filtered_df[filtered_df['rating'].isin(rating_values)]
        else:
            filtered_df = filtered_df[filtered_df['rating'] == rating_values]
        print(f"  ⭐ Rating filter: kept {len(filtered_df)} records")
    
    # Exclude augmented samples
    if conditions.get('exclude_augmented', False):
        filtered_df = filtered_df[filtered_df['author'] != 'AudioAugmenter']
        print(f"  🚫 Exclude augmented: kept {len(filtered_df)} records")
    
    # Include only specific augmentation types
    if 'augmentation_types' in conditions:
        # This assumes augmented files have the augmentation type in filename
        pattern = '|'.join(conditions['augmentation_types'])
        mask = filtered_df['filename'].str.contains(pattern, na=False, regex=True)
        # Also include non-augmented files unless explicitly excluding them
        if not conditions.get('exclude_original', False):
            mask |= (filtered_df['author'] != 'AudioAugmenter')
        filtered_df = filtered_df[mask]
        print(f"  🎛️ Augmentation type filter: kept {len(filtered_df)} records")
    
    # Filter by author
    if 'author' in conditions:
        authors = conditions['author'] if isinstance(conditions['author'], list) else [conditions['author']]
        filtered_df = filtered_df[filtered_df['author'].isin(authors)]
        print(f"  👤 Author filter: kept {len(filtered_df)} records")
    
    # Balance samples per species
    if 'min_samples_per_species' in conditions or 'max_samples_per_species' in conditions:
        balanced_dfs = []
        for species in filtered_df['primary_label'].unique():
            species_df = filtered_df[filtered_df['primary_label'] == species]
            
            # Apply min constraint
            if 'min_samples_per_species' in conditions:
                if len(species_df) < conditions['min_samples_per_species']:
                    print(f"    ⚠️ Skipping {species}: only {len(species_df)} samples (need {conditions['min_samples_per_species']})")
                    continue
            
            # Apply max constraint
            if 'max_samples_per_species' in conditions:
                if len(species_df) > conditions['max_samples_per_species']:
                    species_df = species_df.sample(n=conditions['max_samples_per_species'], random_state=42)
                    print(f"    🎯 Limited {species} to {conditions['max_samples_per_species']} samples")
            
            balanced_dfs.append(species_df)
        
        if balanced_dfs:
            filtered_df = pd.concat(balanced_dfs, ignore_index=True)
        else:
            filtered_df = pd.DataFrame(columns=filtered_df.columns)
        
        print(f"Species balance filter: kept {len(filtered_df)} records")
    
    # Random sampling
    if 'random_sample' in conditions:
        if conditions['random_sample'] < 1.0:
            n_samples = int(len(filtered_df) * conditions['random_sample'])
            filtered_df = filtered_df.sample(n=n_samples, random_state=42)
            print(f"Random sample filter: kept {len(filtered_df)} records")
    
    return filtered_df


def organize_audio_files(metadata_df: pd.DataFrame, target_dir: str):
    """Copy and organize audio files according to metadata"""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Organizing {len(metadata_df)} audio files to {target_dir}")
    
    copied_count = 0
    missing_count = 0
    
    for _, row in metadata_df.iterrows():
        # Determine source file path
        filename = row['filename']
        
        # Handle different source directories
        possible_sources = [
            Path("birdclef-2023/train_audio") / filename,
            Path("birdclef-2023/augmented/train_audio") / filename,
            Path("birdclef-2023/augmented_full/train_audio") / filename,
            Path("birdclef-2023/augmented_basic/train_audio") / filename
        ]
        
        source_path = None
        for path in possible_sources:
            if path.exists():
                source_path = path
                break
        
        if source_path:
            # Create target directory structure
            target_file_path = target_path / filename
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file if it doesn't exist
            if not target_file_path.exists():
                shutil.copy2(source_path, target_file_path)
                copied_count += 1
        else:
            print(f"    ⚠️ Could not find audio file: {filename}")
            missing_count += 1
    
    print(f"  📋 Copied {copied_count} files, {missing_count} missing files")


# Example usage functions
def create_full_augmentation_suite(input_dir: str = "birdclef-2023/train_audio") -> AudioAugmenter:
    """Create augmenter with all augmentations enabled"""
    return AudioAugmenter(
        mixup=True,
        background_noise=True,
        pitch_shift=True,
        time_shift=True,
        freq_masking=True,
        time_masking=True,
        color_jitter=True,
        random_clip=True,
        output_dir="birdclef-2023/augmented_full"
    )


def create_basic_augmentation_suite(input_dir: str = "birdclef-2023/train_audio") -> AudioAugmenter:
    """Create augmenter with basic augmentations only"""
    return AudioAugmenter(
        mixup=True,
        background_noise=True,
        random_clip=True,
        output_dir="birdclef-2023/augmented_basic"
    )


if __name__ == "__main__":
    # Example usage
    print("BirdCLEF Audio Augmentation Demo")
    
    # Create augmenter with all features
    augmenter = create_full_augmentation_suite()
    
    # Process training data with original metadata
    metadata_path = augmenter.process_directory(
        "birdclef-2023/train_audio",
        metadata_csv="birdclef-2023/train_metadata_with_duration.csv"
    )
    
    if metadata_path:
        print(f"\nAugmentation complete!")
        print(f"Augmented files: birdclef-2023/augmented_full/train_audio/")
        print(f"Metadata: {metadata_path}")
    else:
        print("Augmentation failed")