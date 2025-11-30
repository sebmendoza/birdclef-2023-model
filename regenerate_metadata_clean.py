#!/usr/bin/env python3
"""
Update BirdCLEF Augmented Metadata with Proper Inheritance
=========================================================

This script updates the existing augmented_metadata.csv file to properly inherit
latitude, longitude, scientific_name, and common_name from the original files.

NO audio re-processing is done - only the CSV metadata is updated.

Usage:
    python regenerate_metadata.py

Requirements:
    - birdclef-2023/train_metadata_with_duration.csv (original metadata)
    - birdclef-2023/augmented/augmented_metadata.csv (existing augmented metadata)
"""

import pandas as pd
from pathlib import Path


def regenerate_augmented_metadata():
    """Update existing augmented metadata with proper inheritance (no re-augmentation)"""
    
    print("🔄 Updating augmented metadata with proper inheritance...")
    
    # Paths
    original_metadata_path = "birdclef-2023/train_metadata_cleaned.csv"
    augmented_metadata_path = "birdclef-2023/augmented/augmented_metadata.csv"
    
    # Check if files exist
    if not Path(original_metadata_path).exists():
        print(f"❌ Original metadata file not found: {original_metadata_path}")
        return
    
    if not Path(augmented_metadata_path).exists():
        print(f"❌ Augmented metadata file not found: {augmented_metadata_path}")
        print("   Run augmentation first to create the augmented data.")
        return
    
    try:
        # Load metadata files
        print(f"📊 Loading original metadata from {original_metadata_path}")
        orig_df = pd.read_csv(original_metadata_path)
        
        print(f"📊 Loading augmented metadata from {augmented_metadata_path}")
        aug_df = pd.read_csv(augmented_metadata_path)
        
        print(f"   Original records: {len(orig_df)}")
        print(f"   Augmented records: {len(aug_df)}")
        
        # Create lookup dictionary for original metadata: filename -> metadata
        print("🔍 Creating filename lookup dictionary...")
        original_lookup = {}
        for _, row in orig_df.iterrows():
            filename = Path(row['filename']).name  # Extract just filename (e.g., XC113914.ogg)
            original_lookup[filename] = {
                'latitude': row.get('latitude', ''),
                'longitude': row.get('longitude', ''),
                'type': row.get('type', ''),
                'rating': row.get('rating', ''),
                'has_coordinates': row.get('has_coordinates', False),
                'country': row.get('country', ''),
                'continent': row.get('continent', '')
            }
        
        print(f"   Created lookup for {len(original_lookup)} original files")
        
        # Update augmented metadata
        print("🔄 Updating augmented metadata records...")
        updated_count = 0
        mixup_count = 0
        
        for idx, row in aug_df.iterrows():
            filename = row['filename']
            
            # Handle mixup files (inherit from primary file)
            if 'mixup/' in filename:
                mixup_filename = Path(filename).name  # e.g., XC113914_XC129647_mixup_0.651.ogg
                
                # Extract the first XC ID (primary file)
                parts = mixup_filename.split('_')
                if len(parts) >= 2 and parts[0].startswith('XC'):
                    primary_file = parts[0] + '.ogg'  # e.g., XC113914.ogg
                    secondary_file = parts[1] + '.ogg'  # e.g., XC129647.ogg
                    
                    primary_metadata = original_lookup.get(primary_file, {})
                    secondary_metadata = original_lookup.get(secondary_file, {})
                    
                    if primary_metadata:
                        # Update with primary file metadata (cleaned schema)
                        aug_df.at[idx, 'latitude'] = primary_metadata.get('latitude', '')
                        aug_df.at[idx, 'longitude'] = primary_metadata.get('longitude', '')
                        aug_df.at[idx, 'type'] = primary_metadata.get('type', "['mixup']")
                        aug_df.at[idx, 'rating'] = primary_metadata.get('rating', 'synthetic')
                        aug_df.at[idx, 'country'] = primary_metadata.get('country', '')
                        aug_df.at[idx, 'continent'] = primary_metadata.get('continent', '')
                        
                        # Set has_coordinates based on lat/lon availability
                        lat = primary_metadata.get('latitude', '')
                        lon = primary_metadata.get('longitude', '')
                        has_coords = bool(lat and lon and lat != '' and lon != '')
                        aug_df.at[idx, 'has_coordinates'] = has_coords
                        
                        updated_count += 1
                        mixup_count += 1
            
            # Handle regular augmented files (inherit from source file)
            else:
                # Extract original filename from augmented filename
                # e.g., barswa/XC113914_noise_gaussian.ogg -> XC113914.ogg
                base_filename = Path(filename).name
                
                # Remove augmentation suffixes
                original_filename = None
                if '_noise_' in base_filename:
                    original_filename = base_filename.split('_noise_')[0] + '.ogg'
                elif '_pitch_shift' in base_filename:
                    original_filename = base_filename.split('_pitch_shift')[0] + '.ogg'
                elif '_time_shift' in base_filename:
                    original_filename = base_filename.split('_time_shift')[0] + '.ogg'
                elif '_spec_augment' in base_filename:
                    original_filename = base_filename.split('_spec_augment')[0] + '.ogg'
                elif '_color_jitter' in base_filename:
                    original_filename = base_filename.split('_color_jitter')[0] + '.ogg'
                elif '_random_clip' in base_filename:
                    original_filename = base_filename.split('_random_clip')[0] + '.ogg'
                
                if original_filename and original_filename in original_lookup:
                    source_metadata = original_lookup[original_filename]
                    
                    # Update with source file metadata (cleaned schema)
                    aug_df.at[idx, 'latitude'] = source_metadata.get('latitude', '')
                    aug_df.at[idx, 'longitude'] = source_metadata.get('longitude', '')
                    aug_df.at[idx, 'type'] = source_metadata.get('type', "['augmented']")
                    aug_df.at[idx, 'rating'] = source_metadata.get('rating', 'synthetic')
                    aug_df.at[idx, 'country'] = source_metadata.get('country', '')
                    aug_df.at[idx, 'continent'] = source_metadata.get('continent', '')
                    
                    # Set has_coordinates based on lat/lon availability
                    lat = source_metadata.get('latitude', '')
                    lon = source_metadata.get('longitude', '')
                    has_coords = bool(lat and lon and lat != '' and lon != '')
                    aug_df.at[idx, 'has_coordinates'] = has_coords
                    
                    # Copy over type and rating from original file
                    aug_df.at[idx, 'type'] = source_metadata.get('type', "['augmented']")
                    aug_df.at[idx, 'rating'] = source_metadata.get('rating', 'synthetic')
                    
                    updated_count += 1
        
        # Save updated metadata
        print(f"💾 Saving updated metadata to {augmented_metadata_path}")
        aug_df.to_csv(augmented_metadata_path, index=False)
        
        print(f"\n🎉 Successfully updated metadata!")
        print(f"   Updated {updated_count} records ({mixup_count} mixup files)")
        
        # Show metadata quality (cleaned schema)
        non_empty_lat = aug_df['latitude'].fillna('').astype(str).str.strip().ne('').sum()
        has_coordinates = aug_df['has_coordinates'].fillna(False).sum()
        non_empty_country = aug_df['country'].fillna('').astype(str).str.strip().ne('').sum()
        
        print(f"\n📈 Metadata quality check:")
        print(f"   Total samples: {len(aug_df)}")
        print(f"   Files with latitude: {non_empty_lat}/{len(aug_df)} ({non_empty_lat/len(aug_df)*100:.1f}%)")
        print(f"   Files with coordinates: {has_coordinates}/{len(aug_df)} ({has_coordinates/len(aug_df)*100:.1f}%)")
        print(f"   Files with country: {non_empty_country}/{len(aug_df)} ({non_empty_country/len(aug_df)*100:.1f}%)")
        
        # Show species breakdown
        print(f"\n🐦 Species distribution:")
        species_counts = aug_df['primary_label'].value_counts()
        for species, count in species_counts.items():
            print(f"   {species}: {count} samples")
            
    except Exception as e:
        print(f"❌ Error during metadata update: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    regenerate_augmented_metadata()