# """
# Multi-threaded GT Matches Analyzer for SphereCraft Dataset
# Analyzes ground truth match counts and bins pairs into difficulty categories
# Modified to create 3 bins (Easy, Medium, Hard) with simple JSON format
# """

# import numpy as np
# from pathlib import Path
# import re
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from collections import defaultdict
# import argparse
# from tqdm import tqdm
# import json
# import matplotlib.pyplot as plt


# def parse_filename(filename):
#     """
#     Parse filename like 'berlin_00000006_00000176.npz'
#     Returns: (scene_name, frame1, frame2)
#     """
#     stem = Path(filename).stem
    
#     # Pattern: {scene_name}_{frame1}_{frame2}
#     pattern = r'(.+?)_(\d+)_(\d+)$'
#     match = re.match(pattern, stem)
    
#     if not match:
#         return None
    
#     scene_name = match.group(1)
#     frame1 = int(match.group(2))
#     frame2 = int(match.group(3))
    
#     return {
#         'scene': scene_name,
#         'frame1': frame1,
#         'frame2': frame2,
#     }


# def process_file(file_path):
#     """
#     Process a single .npz file and extract GT match count
#     """
#     try:
#         # Parse filename
#         info = parse_filename(file_path.name)
        
#         if not info:
#             return None
        
#         # Load .npz file
#         data = np.load(str(file_path))
        
#         # Count valid GT matches (where gt_matches0 >= 0)
#         if 'gt_matches0' in data:
#             gt_matches = data['gt_matches0']
#             num_matches = (gt_matches >= 0).sum()
#         elif 'gt_matches1' in data:
#             gt_matches = data['gt_matches1']
#             num_matches = (gt_matches >= 0).sum()
#         else:
#             return None
        
#         # Also get frame gap for correlation analysis
#         frame_gap = abs(info['frame2'] - info['frame1'])
        
#         return {
#             'filename': file_path.name,
#             'path': str(file_path),
#             'scene': info['scene'],
#             'frame1': info['frame1'],
#             'frame2': info['frame2'],
#             'frame_gap': frame_gap,
#             'num_matches': int(num_matches),
#         }
        
#     except Exception as e:
#         print(f"Error processing {file_path.name}: {e}")
#         return None


# def process_file_batch(files):
#     """Process a batch of files"""
#     results = []
#     for file_path in files:
#         result = process_file(file_path)
#         if result:
#             results.append(result)
#     return results


# def analyze_gt_matches(data_dir, num_threads=8, exclude_very_hard=True, min_matches_threshold=150):
#     """
#     Analyze GT match counts in dataset using multiple threads
    
#     Args:
#         data_dir: Path to directory containing .npz files
#         num_threads: Number of threads for parallel processing
#         exclude_very_hard: If True, exclude pairs with < min_matches_threshold
#         min_matches_threshold: Minimum number of matches to include a pair (default: 150)
#     """
#     print("\n" + "="*70)
#     print("GROUND TRUTH MATCH COUNT ANALYSIS - 3 BINS")
#     print("="*70)
    
#     data_path = Path(data_dir)
#     if not data_path.exists():
#         print(f"❌ Error: Directory {data_dir} does not exist!")
#         return None
    
#     # Get all .npz files
#     all_files = list(data_path.glob("*.npz"))
#     total_files = len(all_files)
    
#     print(f"\n[1] SCANNING DATASET:")
#     print(f"  Directory: {data_dir}")
#     print(f"  Total .npz files found: {total_files:,}")
    
#     if total_files == 0:
#         print("❌ No .npz files found!")
#         return None
    
#     # Split files into batches for threading
#     batch_size = max(1, total_files // (num_threads * 4))
#     file_batches = [all_files[i:i + batch_size] 
#                     for i in range(0, total_files, batch_size)]
    
#     print(f"  Using {num_threads} threads with {len(file_batches)} batches")
#     print(f"  ⚠️  This will load all .npz files - may take a few minutes...")
    
#     # Process files in parallel
#     all_results = []
    
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = [executor.submit(process_file_batch, batch) 
#                    for batch in file_batches]
        
#         with tqdm(total=len(futures), desc="Processing batches") as pbar:
#             for future in as_completed(futures):
#                 results = future.result()
#                 all_results.extend(results)
#                 pbar.update(1)
    
#     print(f"  Successfully processed: {len(all_results):,}/{total_files:,} files")
    
#     if len(all_results) == 0:
#         print("❌ No files could be processed!")
#         return None
    
#     # Filter out very hard pairs if requested
#     if exclude_very_hard:
#         original_count = len(all_results)
#         all_results = [r for r in all_results if r['num_matches'] >= min_matches_threshold]
#         excluded_count = original_count - len(all_results)
#         print(f"\n  ⚠️  Excluded {excluded_count:,} pairs with < {min_matches_threshold} matches")
#         print(f"  Remaining pairs: {len(all_results):,}")
    
#     # Extract match counts
#     match_counts = np.array([r['num_matches'] for r in all_results])
#     frame_gaps = np.array([r['frame_gap'] for r in all_results])
    
#     # Statistics
#     print(f"\n[2] GT MATCH COUNT STATISTICS:")
#     print(f"  Min matches: {match_counts.min()}")
#     print(f"  Max matches: {match_counts.max()}")
#     print(f"  Mean matches: {match_counts.mean():.2f}")
#     print(f"  Median matches: {np.median(match_counts):.2f}")
#     print(f"  Std matches: {match_counts.std():.2f}")
    
#     # Percentiles for bin design
#     percentiles = [25, 33, 50, 66, 75, 90]
#     print(f"\n  Percentiles:")
#     for p in percentiles:
#         val = np.percentile(match_counts, p)
#         print(f"    {p:2d}th percentile: {val:.0f} matches")
    
#     # Correlation analysis
#     correlation = np.corrcoef(match_counts, frame_gaps)[0, 1]
#     print(f"\n  Correlation between frame gap and match count: {correlation:.3f}")
#     if correlation < -0.5:
#         print(f"    → Strong negative correlation: larger gap = fewer matches ✓")
#     elif correlation < -0.3:
#         print(f"    → Moderate negative correlation")
#     else:
#         print(f"    → Weak correlation")
    
#     # Define 3 bins based on percentiles (33rd and 66th)
#     # This creates roughly equal-sized bins
#     p33 = np.percentile(match_counts, 33)
#     p66 = np.percentile(match_counts, 66)
    
#     bins_config = [
#         {'name': 'easy', 'min': p66, 'max': float('inf'), 'color': 'green'},
#         {'name': 'medium', 'min': p33, 'max': p66, 'color': 'yellow'},
#         {'name': 'hard', 'min': min_matches_threshold if exclude_very_hard else 0, 
#          'max': p33, 'color': 'orange'},
#     ]
    
#     print(f"\n[3] BIN THRESHOLDS (3 bins):")
#     print(f"  Easy:   {bins_config[0]['min']:.0f}+ matches")
#     print(f"  Medium: {bins_config[1]['min']:.0f}-{bins_config[1]['max']:.0f} matches")
#     print(f"  Hard:   {bins_config[2]['min']:.0f}-{bins_config[2]['max']:.0f} matches")
#     if exclude_very_hard:
#         print(f"  (Excluded: <{min_matches_threshold} matches)")
    
#     # Bin the data
#     binned_data = {bin_cfg['name']: [] for bin_cfg in bins_config}
    
#     for result in all_results:
#         num_matches = result['num_matches']
#         for bin_cfg in bins_config:
#             if bin_cfg['min'] <= num_matches < bin_cfg['max']:
#                 binned_data[bin_cfg['name']].append(result)
#                 break
    
#     # Print bin statistics
#     print(f"\n[4] BINNED DISTRIBUTION:")
#     total = len(all_results)
    
#     scene_counts = defaultdict(lambda: {'easy': 0, 'medium': 0, 'hard': 0})
    
#     for bin_cfg in bins_config:
#         bin_name = bin_cfg['name']
#         count = len(binned_data[bin_name])
#         pct = 100 * count / total if total > 0 else 0
        
#         if bin_cfg['max'] == float('inf'):
#             range_str = f"[{bin_cfg['min']:.0f}, ∞)"
#         else:
#             range_str = f"[{bin_cfg['min']:.0f}, {bin_cfg['max']:.0f})"
        
#         # Calculate average match count and frame gap for this bin
#         if count > 0:
#             bin_matches = [r['num_matches'] for r in binned_data[bin_name]]
#             bin_gaps = [r['frame_gap'] for r in binned_data[bin_name]]
#             avg_matches = np.mean(bin_matches)
#             avg_gap = np.mean(bin_gaps)
            
#             print(f"  {bin_name.capitalize():8s} {range_str:15s}: {count:7,} pairs ({pct:5.1f}%)")
#             print(f"           Avg matches: {avg_matches:6.1f}  |  Avg frame gap: {avg_gap:6.1f}")
            
#             # Track by scene
#             for result in binned_data[bin_name]:
#                 scene_counts[result['scene']][bin_name] += 1
#         else:
#             print(f"  {bin_name.capitalize():8s} {range_str:15s}: {count:7,} pairs ({pct:5.1f}%)")
    
#     # Scene breakdown
#     if len(scene_counts) > 0:
#         print(f"\n[5] BREAKDOWN BY SCENE:")
#         for scene, counts in sorted(scene_counts.items()):
#             total_scene = sum(counts.values())
#             print(f"  {scene:15s}: {total_scene:6,} pairs")
#             print(f"    Easy: {counts['easy']:5,} | Medium: {counts['medium']:5,} | Hard: {counts['hard']:5,}")
    
#     # Curriculum learning suggestions
#     print(f"\n[6] CURRICULUM LEARNING SUGGESTIONS:")
#     print(f"  Phase 1 (Epochs 0-10): Easy only")
#     print(f"    → {len(binned_data['easy']):,} pairs")
#     print(f"  Phase 2 (Epochs 10-20): Easy + Medium")
#     print(f"    → {len(binned_data['easy']) + len(binned_data['medium']):,} pairs")
#     print(f"  Phase 3 (Epochs 20-40): All data")
#     print(f"    → {total:,} pairs")
    
#     print("="*70 + "\n")
    
#     return {
#         'total_pairs': total,
#         'match_counts': match_counts,
#         'frame_gaps': frame_gaps,
#         'binned_data': binned_data,
#         'bins_config': bins_config,
#         'scene_counts': dict(scene_counts),
#         'all_results': all_results,
#         'correlation': correlation,
#         'excluded_very_hard': exclude_very_hard,
#         'min_matches_threshold': min_matches_threshold,
#     }


# def create_visualizations(match_counts, frame_gaps, binned_data, bins_config, output_dir):
#     """Create visualization plots"""
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     # 1. Histogram of match counts
#     axes[0, 0].hist(match_counts, bins=50, edgecolor='black', alpha=0.7)
#     axes[0, 0].set_xlabel('Number of GT Matches')
#     axes[0, 0].set_ylabel('Count')
#     axes[0, 0].set_title('Distribution of GT Match Counts')
#     axes[0, 0].grid(alpha=0.3)
    
#     # Add bin boundaries
#     for i, bin_cfg in enumerate(bins_config[:-1]):  # Skip last (easy) boundary
#         axes[0, 0].axvline(bins_config[i+1]['min'], color='red', 
#                           linestyle='--', alpha=0.7, linewidth=2,
#                           label=f"{bins_config[i+1]['name']} threshold")
#     axes[0, 0].legend()
    
#     # 2. Bin distribution (bar chart)
#     bin_names = [cfg['name'].capitalize() for cfg in bins_config]
#     bin_counts = [len(binned_data[cfg['name']]) for cfg in bins_config]
#     bin_colors = [cfg['color'] for cfg in bins_config]
    
#     bars = axes[0, 1].bar(bin_names, bin_counts, color=bin_colors, 
#                           edgecolor='black', alpha=0.7, linewidth=2)
#     axes[0, 1].set_ylabel('Number of Pairs')
#     axes[0, 1].set_title('Pairs per Difficulty Bin (3 Bins)')
#     axes[0, 1].grid(alpha=0.3, axis='y')
    
#     # Add count labels on bars
#     for bar, count in zip(bars, bin_counts):
#         height = bar.get_height()
#         axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
#                        f'{count:,}',
#                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
#     # 3. Scatter: Frame gap vs Match count
#     sample_size = min(10000, len(match_counts))
#     indices = np.random.choice(len(match_counts), sample_size, replace=False)
#     axes[1, 0].scatter(frame_gaps[indices], match_counts[indices], 
#                       alpha=0.3, s=10, c='blue')
#     axes[1, 0].set_xlabel('Frame Gap')
#     axes[1, 0].set_ylabel('Number of GT Matches')
#     axes[1, 0].set_title('Frame Gap vs GT Matches')
#     axes[1, 0].grid(alpha=0.3)
    
#     # Add correlation text
#     correlation = np.corrcoef(match_counts, frame_gaps)[0, 1]
#     axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
#                    transform=axes[1, 0].transAxes,
#                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#                    verticalalignment='top', fontsize=10)
    
#     # 4. Average matches per bin
#     bin_avg_matches = []
#     for cfg in bins_config:
#         if len(binned_data[cfg['name']]) > 0:
#             matches = [r['num_matches'] for r in binned_data[cfg['name']]]
#             bin_avg_matches.append(np.mean(matches))
#         else:
#             bin_avg_matches.append(0)
    
#     bars = axes[1, 1].bar(bin_names, bin_avg_matches, color=bin_colors,
#                           edgecolor='black', alpha=0.7, linewidth=2)
#     axes[1, 1].set_ylabel('Average GT Matches')
#     axes[1, 1].set_title('Average Matches per Bin')
#     axes[1, 1].grid(alpha=0.3, axis='y')
    
#     # Add labels
#     for bar, avg in zip(bars, bin_avg_matches):
#         height = bar.get_height()
#         axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
#                        f'{avg:.0f}',
#                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
#     plt.tight_layout()
    
#     output_path = Path(output_dir) / 'gt_match_analysis_3bins.png'
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     print(f"  ✓ Saved visualization to {output_path}")
#     plt.close()


# def save_binned_lists(binned_data, bins_config, output_dir):
#     """
#     Save SIMPLE lists of files for each bin (just filenames, no metadata)
#     This format is compatible with curriculum learning code
#     """
#     output_dir = Path(output_dir)
    
#     print(f"\n[7] SAVING BIN FILES:")
    
#     for bin_cfg in bins_config:
#         bin_name = bin_cfg['name']
#         data = binned_data[bin_name]
        
#         # Create SIMPLE list of just filenames
#         filenames = [item['filename'] for item in data]
        
#         # Save as JSON (simple list)
#         filename = f"bin_{bin_name}_by_matches.json"
#         output_file = output_dir / filename
        
#         with open(output_file, 'w') as f:
#             json.dump(filenames, f, indent=2)
        
#         print(f"  ✓ Saved {bin_name.capitalize():8s} → {filename:30s} ({len(data):6,} pairs)")
    
#     # Also save a detailed summary with metadata
#     summary_file = output_dir / 'binning_summary_3bins.json'
#     summary = {
#         'binning_method': 'gt_match_count',
#         'num_bins': 3,
#         'total_pairs': sum(len(binned_data[cfg['name']]) for cfg in bins_config),
#         'bins': [
#             {
#                 'name': cfg['name'],
#                 'min_matches': cfg['min'],
#                 'max_matches': cfg['max'] if cfg['max'] != float('inf') else 'inf',
#                 'count': len(binned_data[cfg['name']]),
#                 'percentage': 100 * len(binned_data[cfg['name']]) / 
#                              sum(len(binned_data[c['name']]) for c in bins_config),
#                 'files': [item['filename'] for item in binned_data[cfg['name']]]
#             }
#             for cfg in bins_config
#         ]
#     }
    
#     with open(summary_file, 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     print(f"  ✓ Saved summary → {summary_file.name}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Analyze GT matches and create 3 bins')
#     parser.add_argument('--data_dir', type=str, 
#                        default="/data/code/glue-factory/data/finetuning/finetuning_pairs_spherecraft",
#                        help='Directory containing .npz files')
#     parser.add_argument('--threads', type=int, default=8,
#                        help='Number of threads for parallel processing')
#     parser.add_argument('--exclude_very_hard', action='store_true', default=True,
#                        help='Exclude pairs with very few matches (default: True)')
#     parser.add_argument('--include_very_hard', action='store_true',
#                        help='Include very hard pairs (overrides --exclude_very_hard)')
#     parser.add_argument('--min_matches', type=int, default=150,
#                        help='Minimum matches threshold for exclusion (default: 150)')
    
#     args = parser.parse_args()
    
#     # Handle the include/exclude logic
#     exclude_very_hard = args.exclude_very_hard and not args.include_very_hard
    
#     print(f"\nConfiguration:")
#     print(f"  Data directory: {args.data_dir}")
#     print(f"  Threads: {args.threads}")
#     print(f"  Exclude very hard pairs: {exclude_very_hard}")
#     if exclude_very_hard:
#         print(f"  Min matches threshold: {args.min_matches}")
    
#     # Run analysis
#     results = analyze_gt_matches(
#         args.data_dir, 
#         num_threads=args.threads,
#         exclude_very_hard=exclude_very_hard,
#         min_matches_threshold=args.min_matches
#     )
    
#     if results:
#         output_dir = Path(args.data_dir).parent
        
#         # Save bin files
#         save_binned_lists(
#             results['binned_data'], 
#             results['bins_config'], 
#             output_dir
#         )
        
#         # Create visualizations
#         print(f"\n[8] CREATING VISUALIZATIONS:")
#         create_visualizations(
#             results['match_counts'],
#             results['frame_gaps'],
#             results['binned_data'],
#             results['bins_config'],
#             output_dir
#         )
        
#         print("\n✅ Analysis complete!")
#         print(f"\n📁 Output files saved to: {output_dir}")
#         print(f"   • bin_easy_by_matches.json")
#         print(f"   • bin_medium_by_matches.json")
#         print(f"   • bin_hard_by_matches.json")
#         print(f"   • binning_summary_3bins.json")
#         print(f"   • gt_match_analysis_3bins.png")
#         print(f"\n💡 Use these files for curriculum learning!")

# """
# RUN:

# python3 gluefactory/scripts/create_bins.py --data_dir /data/code/glue-factory/data/finetuning/finetuning_pairs_spherecraft --threads 8 --min_matches 150
# """

"""
Multi-threaded GT Matches Analyzer for SphereCraft and Real Scene Datasets
Analyzes ground truth match counts and bins pairs into difficulty categories
Modified to create 3 bins (Easy, Medium, Hard) with simple JSON format

Supports two dataset types:
- SphereCraft: berlin_00000006_00000176.npz (with frame gap analysis)
- Real Scenes: 6890857f0986c3000169f148_6890857f0986c3000169f149.npz (no frame gap)
"""
import numpy as np
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


def parse_filename_spherecraft(filename):
    """
    Parse SphereCraft filename like 'berlin_00000006_00000176.npz'
    Returns: (scene_name, frame1, frame2)
    """
    stem = Path(filename).stem
    
    # Pattern: {scene_name}_{frame1}_{frame2}
    pattern = r'(.+?)_(\d+)_(\d+)$'
    match = re.match(pattern, stem)
    
    if not match:
        return None
    
    scene_name = match.group(1)
    frame1 = int(match.group(2))
    frame2 = int(match.group(3))
    
    return {
        'scene': scene_name,
        'frame1': frame1,
        'frame2': frame2,
        'pair_id': f"{scene_name}_{frame1}_{frame2}"
    }


def parse_filename_real(filename):
    """
    Parse Real Scene filename like '6890857f0986c3000169f148_6890857f0986c3000169f149.npz'
    Returns: (id1, id2)
    """
    stem = Path(filename).stem
    
    # Pattern: {id1}_{id2}
    parts = stem.split('_')
    
    if len(parts) != 2:
        return None
    
    id1, id2 = parts
    
    return {
        'scene': 'real_scene',  # Generic scene name for real scenes
        'id1': id1,
        'id2': id2,
        'pair_id': stem
    }


def process_file(file_path, is_real=False):
    """
    Process a single .npz file and extract GT match count
    
    Args:
        file_path: Path to .npz file
        is_real: If True, use real scene naming convention (no frame gap)
    """
    try:
        # Parse filename based on dataset type
        if is_real:
            info = parse_filename_real(file_path.name)
        else:
            info = parse_filename_spherecraft(file_path.name)
        
        if not info:
            return None
        
        # Load .npz file
        data = np.load(str(file_path))
        
        # Count valid GT matches (where gt_matches0 >= 0)
        if 'gt_matches0' in data:
            gt_matches = data['gt_matches0']
            num_matches = (gt_matches >= 0).sum()
        elif 'gt_matches1' in data:
            gt_matches = data['gt_matches1']
            num_matches = (gt_matches >= 0).sum()
        else:
            return None
        
        result = {
            'filename': file_path.name,
            'path': str(file_path),
            'scene': info['scene'],
            'pair_id': info['pair_id'],
            'num_matches': int(num_matches),
        }
        
        # Add frame gap only for SphereCraft
        if not is_real:
            frame_gap = abs(info['frame2'] - info['frame1'])
            result['frame1'] = info['frame1']
            result['frame2'] = info['frame2']
            result['frame_gap'] = frame_gap
        else:
            result['id1'] = info['id1']
            result['id2'] = info['id2']
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None


def process_file_batch(files, is_real=False):
    """Process a batch of files"""
    results = []
    for file_path in files:
        result = process_file(file_path, is_real=is_real)
        if result:
            results.append(result)
    return results


def analyze_gt_matches(data_dir, num_threads=8, exclude_very_hard=True, 
                       min_matches_threshold=150, is_real=False):
    """
    Analyze GT match counts in dataset using multiple threads
    
    Args:
        data_dir: Path to directory containing .npz files
        num_threads: Number of threads for parallel processing
        exclude_very_hard: If True, exclude pairs with < min_matches_threshold
        min_matches_threshold: Minimum number of matches to include a pair
        is_real: If True, dataset is real scenes (different naming convention)
    """
    dataset_type = "REAL SCENES" if is_real else "SPHERECRAFT"
    
    print("\n" + "="*70)
    print(f"GROUND TRUTH MATCH COUNT ANALYSIS - 3 BINS ({dataset_type})")
    print("="*70)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Error: Directory {data_dir} does not exist!")
        return None
    
    # Get all .npz files
    all_files = list(data_path.glob("*.npz"))
    total_files = len(all_files)
    
    print(f"\n[1] SCANNING DATASET:")
    print(f"  Directory: {data_dir}")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Total .npz files found: {total_files:,}")
    
    if total_files == 0:
        print("❌ No .npz files found!")
        return None
    
    # Split files into batches for threading
    batch_size = max(1, total_files // (num_threads * 4))
    file_batches = [all_files[i:i + batch_size] 
                    for i in range(0, total_files, batch_size)]
    
    print(f"  Using {num_threads} threads with {len(file_batches)} batches")
    print(f"  ⚠️  This will load all .npz files - may take a few minutes...")
    
    # Process files in parallel
    all_results = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_file_batch, batch, is_real=is_real) 
                   for batch in file_batches]
        
        with tqdm(total=len(futures), desc="Processing batches") as pbar:
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
                pbar.update(1)
    
    print(f"  Successfully processed: {len(all_results):,}/{total_files:,} files")
    
    if len(all_results) == 0:
        print("❌ No files could be processed!")
        return None
    
    # Filter out very hard pairs if requested
    if exclude_very_hard:
        original_count = len(all_results)
        all_results = [r for r in all_results if r['num_matches'] >= min_matches_threshold]
        excluded_count = original_count - len(all_results)
        print(f"\n  ⚠️  Excluded {excluded_count:,} pairs with < {min_matches_threshold} matches")
        print(f"  Remaining pairs: {len(all_results):,}")
    
    # Extract match counts
    match_counts = np.array([r['num_matches'] for r in all_results])
    
    # Frame gaps only for SphereCraft
    frame_gaps = None
    if not is_real:
        frame_gaps = np.array([r['frame_gap'] for r in all_results])
    
    # Statistics
    print(f"\n[2] GT MATCH COUNT STATISTICS:")
    print(f"  Min matches: {match_counts.min()}")
    print(f"  Max matches: {match_counts.max()}")
    print(f"  Mean matches: {match_counts.mean():.2f}")
    print(f"  Median matches: {np.median(match_counts):.2f}")
    print(f"  Std matches: {match_counts.std():.2f}")
    
    # Percentiles for bin design
    percentiles = [25, 33, 50, 66, 75, 90]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(match_counts, p)
        print(f"    {p:2d}th percentile: {val:.0f} matches")
    
    # Correlation analysis (only for SphereCraft)
    correlation = None
    if not is_real:
        correlation = np.corrcoef(match_counts, frame_gaps)[0, 1]
        print(f"\n  Correlation between frame gap and match count: {correlation:.3f}")
        if correlation < -0.5:
            print(f"    → Strong negative correlation: larger gap = fewer matches ✓")
        elif correlation < -0.3:
            print(f"    → Moderate negative correlation")
        else:
            print(f"    → Weak correlation")
    
    # Define 3 bins based on percentiles (33rd and 66th)
    p33 = np.percentile(match_counts, 33)
    p66 = np.percentile(match_counts, 66)
    
    bins_config = [
        {'name': 'easy', 'min': p66, 'max': float('inf'), 'color': 'green'},
        {'name': 'medium', 'min': p33, 'max': p66, 'color': 'yellow'},
        {'name': 'hard', 'min': min_matches_threshold if exclude_very_hard else 0, 
         'max': p33, 'color': 'orange'},
    ]
    
    print(f"\n[3] BIN THRESHOLDS (3 bins):")
    print(f"  Easy:   {bins_config[0]['min']:.0f}+ matches")
    print(f"  Medium: {bins_config[1]['min']:.0f}-{bins_config[1]['max']:.0f} matches")
    print(f"  Hard:   {bins_config[2]['min']:.0f}-{bins_config[2]['max']:.0f} matches")
    if exclude_very_hard:
        print(f"  (Excluded: <{min_matches_threshold} matches)")
    
    # Bin the data
    binned_data = {bin_cfg['name']: [] for bin_cfg in bins_config}
    
    for result in all_results:
        num_matches = result['num_matches']
        for bin_cfg in bins_config:
            if bin_cfg['min'] <= num_matches < bin_cfg['max']:
                binned_data[bin_cfg['name']].append(result)
                break
    
    # Print bin statistics
    print(f"\n[4] BINNED DISTRIBUTION:")
    total = len(all_results)
    
    scene_counts = defaultdict(lambda: {'easy': 0, 'medium': 0, 'hard': 0})
    
    for bin_cfg in bins_config:
        bin_name = bin_cfg['name']
        count = len(binned_data[bin_name])
        pct = 100 * count / total if total > 0 else 0
        
        if bin_cfg['max'] == float('inf'):
            range_str = f"[{bin_cfg['min']:.0f}, ∞)"
        else:
            range_str = f"[{bin_cfg['min']:.0f}, {bin_cfg['max']:.0f})"
        
        # Calculate average match count (and frame gap for SphereCraft)
        if count > 0:
            bin_matches = [r['num_matches'] for r in binned_data[bin_name]]
            avg_matches = np.mean(bin_matches)
            
            if not is_real:
                bin_gaps = [r['frame_gap'] for r in binned_data[bin_name]]
                avg_gap = np.mean(bin_gaps)
                print(f"  {bin_name.capitalize():8s} {range_str:15s}: {count:7,} pairs ({pct:5.1f}%)")
                print(f"           Avg matches: {avg_matches:6.1f}  |  Avg frame gap: {avg_gap:6.1f}")
            else:
                print(f"  {bin_name.capitalize():8s} {range_str:15s}: {count:7,} pairs ({pct:5.1f}%)")
                print(f"           Avg matches: {avg_matches:6.1f}")
            
            # Track by scene
            for result in binned_data[bin_name]:
                scene_counts[result['scene']][bin_name] += 1
        else:
            print(f"  {bin_name.capitalize():8s} {range_str:15s}: {count:7,} pairs ({pct:5.1f}%)")
    
    # Scene breakdown
    if len(scene_counts) > 0 and not is_real:
        print(f"\n[5] BREAKDOWN BY SCENE:")
        for scene, counts in sorted(scene_counts.items()):
            total_scene = sum(counts.values())
            print(f"  {scene:15s}: {total_scene:6,} pairs")
            print(f"    Easy: {counts['easy']:5,} | Medium: {counts['medium']:5,} | Hard: {counts['hard']:5,}")
    
    # Curriculum learning suggestions
    section_num = 6 if not is_real else 5
    print(f"\n[{section_num}] CURRICULUM LEARNING SUGGESTIONS:")
    print(f"  Phase 1 (Epochs 0-10): Easy only")
    print(f"    → {len(binned_data['easy']):,} pairs")
    print(f"  Phase 2 (Epochs 10-20): Easy + Medium")
    print(f"    → {len(binned_data['easy']) + len(binned_data['medium']):,} pairs")
    print(f"  Phase 3 (Epochs 20-40): All data")
    print(f"    → {total:,} pairs")
    
    print("="*70 + "\n")
    
    return {
        'total_pairs': total,
        'match_counts': match_counts,
        'frame_gaps': frame_gaps,
        'binned_data': binned_data,
        'bins_config': bins_config,
        'scene_counts': dict(scene_counts),
        'all_results': all_results,
        'correlation': correlation,
        'excluded_very_hard': exclude_very_hard,
        'min_matches_threshold': min_matches_threshold,
        'is_real': is_real,
    }


def create_visualizations(match_counts, frame_gaps, binned_data, bins_config, 
                         output_dir, is_real=False):
    """Create visualization plots"""
    if is_real:
        # Real scenes: 3 plots (no frame gap)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Histogram of match counts
        axes[0].hist(match_counts, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Number of GT Matches')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of GT Match Counts')
        axes[0].grid(alpha=0.3)
        
        # Add bin boundaries
        for i, bin_cfg in enumerate(bins_config[:-1]):
            axes[0].axvline(bins_config[i+1]['min'], color='red', 
                           linestyle='--', alpha=0.7, linewidth=2,
                           label=f"{bins_config[i+1]['name']} threshold")
        axes[0].legend()
        
        # 2. Bin distribution (bar chart)
        bin_names = [cfg['name'].capitalize() for cfg in bins_config]
        bin_counts = [len(binned_data[cfg['name']]) for cfg in bins_config]
        bin_colors = [cfg['color'] for cfg in bins_config]
        
        bars = axes[1].bar(bin_names, bin_counts, color=bin_colors, 
                          edgecolor='black', alpha=0.7, linewidth=2)
        axes[1].set_ylabel('Number of Pairs')
        axes[1].set_title('Pairs per Difficulty Bin (3 Bins)')
        axes[1].grid(alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{count:,}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Average matches per bin
        bin_avg_matches = []
        for cfg in bins_config:
            if len(binned_data[cfg['name']]) > 0:
                matches = [r['num_matches'] for r in binned_data[cfg['name']]]
                bin_avg_matches.append(np.mean(matches))
            else:
                bin_avg_matches.append(0)
        
        bars = axes[2].bar(bin_names, bin_avg_matches, color=bin_colors,
                          edgecolor='black', alpha=0.7, linewidth=2)
        axes[2].set_ylabel('Average GT Matches')
        axes[2].set_title('Average Matches per Bin')
        axes[2].grid(alpha=0.3, axis='y')
        
        # Add labels
        for bar, avg in zip(bars, bin_avg_matches):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{avg:.0f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        output_filename = 'gt_match_analysis_3bins_real.png'
        
    else:
        # SphereCraft: 4 plots (with frame gap)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram of match counts
        axes[0, 0].hist(match_counts, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Number of GT Matches')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of GT Match Counts')
        axes[0, 0].grid(alpha=0.3)
        
        # Add bin boundaries
        for i, bin_cfg in enumerate(bins_config[:-1]):
            axes[0, 0].axvline(bins_config[i+1]['min'], color='red', 
                              linestyle='--', alpha=0.7, linewidth=2,
                              label=f"{bins_config[i+1]['name']} threshold")
        axes[0, 0].legend()
        
        # 2. Bin distribution (bar chart)
        bin_names = [cfg['name'].capitalize() for cfg in bins_config]
        bin_counts = [len(binned_data[cfg['name']]) for cfg in bins_config]
        bin_colors = [cfg['color'] for cfg in bins_config]
        
        bars = axes[0, 1].bar(bin_names, bin_counts, color=bin_colors, 
                              edgecolor='black', alpha=0.7, linewidth=2)
        axes[0, 1].set_ylabel('Number of Pairs')
        axes[0, 1].set_title('Pairs per Difficulty Bin (3 Bins)')
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{count:,}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Scatter: Frame gap vs Match count
        sample_size = min(10000, len(match_counts))
        indices = np.random.choice(len(match_counts), sample_size, replace=False)
        axes[1, 0].scatter(frame_gaps[indices], match_counts[indices], 
                          alpha=0.3, s=10, c='blue')
        axes[1, 0].set_xlabel('Frame Gap')
        axes[1, 0].set_ylabel('Number of GT Matches')
        axes[1, 0].set_title('Frame Gap vs GT Matches')
        axes[1, 0].grid(alpha=0.3)
        
        # Add correlation text
        correlation = np.corrcoef(match_counts, frame_gaps)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=10)
        
        # 4. Average matches per bin
        bin_avg_matches = []
        for cfg in bins_config:
            if len(binned_data[cfg['name']]) > 0:
                matches = [r['num_matches'] for r in binned_data[cfg['name']]]
                bin_avg_matches.append(np.mean(matches))
            else:
                bin_avg_matches.append(0)
        
        bars = axes[1, 1].bar(bin_names, bin_avg_matches, color=bin_colors,
                              edgecolor='black', alpha=0.7, linewidth=2)
        axes[1, 1].set_ylabel('Average GT Matches')
        axes[1, 1].set_title('Average Matches per Bin')
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        # Add labels
        for bar, avg in zip(bars, bin_avg_matches):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{avg:.0f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        output_filename = 'gt_match_analysis_3bins.png'
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_path}")
    plt.close()


def save_binned_lists(binned_data, bins_config, output_dir, is_real=False):
    """
    Save SIMPLE lists of files for each bin (just filenames, no metadata)
    This format is compatible with curriculum learning code
    """
    output_dir = Path(output_dir)
    
    section_num = 8 if not is_real else 7
    print(f"\n[{section_num}] SAVING BIN FILES:")
    
    suffix = "_real" if is_real else ""
    
    for bin_cfg in bins_config:
        bin_name = bin_cfg['name']
        data = binned_data[bin_name]
        
        # Create SIMPLE list of just filenames
        filenames = [item['filename'] for item in data]
        
        # Save as JSON (simple list)
        filename = f"bin_{bin_name}_by_matches{suffix}.json"
        output_file = output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(filenames, f, indent=2)
        
        print(f"  ✓ Saved {bin_name.capitalize():8s} → {filename:35s} ({len(data):6,} pairs)")
    
    # Also save a detailed summary with metadata
    summary_file = output_dir / f'binning_summary_3bins{suffix}.json'
    summary = {
        'binning_method': 'gt_match_count',
        'dataset_type': 'real_scenes' if is_real else 'spherecraft',
        'num_bins': 3,
        'total_pairs': sum(len(binned_data[cfg['name']]) for cfg in bins_config),
        'bins': [
            {
                'name': cfg['name'],
                'min_matches': cfg['min'],
                'max_matches': cfg['max'] if cfg['max'] != float('inf') else 'inf',
                'count': len(binned_data[cfg['name']]),
                'percentage': 100 * len(binned_data[cfg['name']]) / 
                             sum(len(binned_data[c['name']]) for c in bins_config),
                'files': [item['filename'] for item in binned_data[cfg['name']]]
            }
            for cfg in bins_config
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Saved summary → {summary_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze GT matches and create 3 bins for SphereCraft or Real Scene datasets'
    )
    parser.add_argument('--data_dir', type=str, 
                       default="/data/code/glue-factory/data/finetuning/finetuning_pairs_spherecraft",
                       help='Directory containing .npz files')
    parser.add_argument('--threads', type=int, default=8,
                       help='Number of threads for parallel processing')
    parser.add_argument('--exclude_very_hard', action='store_true', default=True,
                       help='Exclude pairs with very few matches (default: True)')
    parser.add_argument('--include_very_hard', action='store_true',
                       help='Include very hard pairs (overrides --exclude_very_hard)')
    parser.add_argument('--min_matches', type=int, default=150,
                       help='Minimum matches threshold for exclusion (default: 150)')
    parser.add_argument('--real', action='store_true',
                       help='Use Real Scene dataset naming convention (id1_id2.npz) instead of SphereCraft')
    
    args = parser.parse_args()
    
    # Handle the include/exclude logic
    exclude_very_hard = args.exclude_very_hard and not args.include_very_hard
    
    dataset_type = "Real Scenes" if args.real else "SphereCraft"
    
    print(f"\nConfiguration:")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Threads: {args.threads}")
    print(f"  Exclude very hard pairs: {exclude_very_hard}")
    if exclude_very_hard:
        print(f"  Min matches threshold: {args.min_matches}")
    
    # Run analysis
    results = analyze_gt_matches(
        args.data_dir, 
        num_threads=args.threads,
        exclude_very_hard=exclude_very_hard,
        min_matches_threshold=args.min_matches,
        is_real=args.real
    )
    
    if results:
        output_dir = Path(args.data_dir).parent
        
        # Save bin files
        save_binned_lists(
            results['binned_data'], 
            results['bins_config'], 
            output_dir,
            is_real=args.real
        )
        
        # Create visualizations
        section_num = 9 if not args.real else 8
        print(f"\n[{section_num}] CREATING VISUALIZATIONS:")
        create_visualizations(
            results['match_counts'],
            results['frame_gaps'],
            results['binned_data'],
            results['bins_config'],
            output_dir,
            is_real=args.real
        )
        
        suffix = "_real" if args.real else ""
        
        print("\n✅ Analysis complete!")
        print(f"\n📁 Output files saved to: {output_dir}")
        print(f"   • bin_easy_by_matches{suffix}.json")
        print(f"   • bin_medium_by_matches{suffix}.json")
        print(f"   • bin_hard_by_matches{suffix}.json")
        print(f"   • binning_summary_3bins{suffix}.json")
        print(f"   • gt_match_analysis_3bins{'' if not args.real else '_real'}.png")
        print(f"\n💡 Use these files for curriculum learning!")

"""
python3 gluefactory/scripts/create_bins.py \
  --data_dir "/data/code/glue-factory/data/finetuning/finetuning_pairs_1" \
  --threads 20 \
  --min_matches 150 \
  --real
"""