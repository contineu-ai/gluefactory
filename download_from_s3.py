import os
import argparse
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# --- Configuration ---

# The S3 bucket where the data is stored.
BUCKET_NAME = "spherecraft-dataset"

# The local base directory where scenes will be downloaded.
# The final path will be: LOCAL_BASE_DIRECTORY/{scene_name}/
LOCAL_BASE_DIRECTORY = "/data/code/"

# List of S3 prefixes (files or folders) to download for each scene.
# These will be appended to the scene name.
ITEMS_TO_DOWNLOAD = [
    "superpoint_train.txt",
    "images",
    "depthmaps",
    "extr",
    "features_xfeat_spherical",
]

# Number of parallel download threads.
DEFAULT_WORKERS = 10

# --- S3 Download Logic ---

def _download_file(s3_client, bucket, s3_key, local_path):
    """Helper function to download a single file and handle directory creation."""
    try:
        # Ensure the local directory for the file exists.
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket, s3_key, local_path)
        return s3_key
    except Exception as e:
        print(f"Failed to download {s3_key}: {e}")
        return None

def download_scene_data(scene_name: str, num_workers: int):
    """
    Downloads all specified data for a given scene from S3.

    Args:
        scene_name (str): The name of the scene to download (e.g., "berlin").
        num_workers (int): The number of concurrent threads for downloading.
    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')

    print(f"🚀 Starting download for scene: '{scene_name}'")
    print(f"Target directory: {os.path.join(LOCAL_BASE_DIRECTORY, scene_name)}")
    print("-" * 50)

    for item in ITEMS_TO_DOWNLOAD:
        s3_prefix = f"{scene_name}/{item}"
        print(f"\nProcessing: s3://{BUCKET_NAME}/{s3_prefix}")

        try:
            # Step 1: List all objects for the current prefix
            download_tasks = []
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix)
            
            for page in pages:
                if "Contents" not in page:
                    continue
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Skip if it's a folder marker (ends with /)
                    if s3_key.endswith('/'):
                        continue
                    
                    # Construct the full local path, mirroring S3 structure
                    local_file_path = os.path.join(LOCAL_BASE_DIRECTORY, s3_key)
                    download_tasks.append({'s3_key': s3_key, 'local_path': local_file_path})

            if not download_tasks:
                print(f"⚠️ No files found for prefix '{s3_prefix}'. Skipping.")
                continue

            print(f"Found {len(download_tasks)} file(s) to download.")

            # Step 2: Download files concurrently with a progress bar
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Create a future for each download task
                futures = [
                    executor.submit(_download_file, s3_client, BUCKET_NAME, task['s3_key'], task['local_path'])
                    for task in download_tasks
                ]

                # Use tqdm to show a progress bar as futures complete
                failed_count = 0
                for future in tqdm(as_completed(futures), total=len(download_tasks), desc="Downloading", unit="file"):
                    result = future.result()
                    if result is None:
                        failed_count += 1
                
                if failed_count > 0:
                    print(f"⚠️ {failed_count} file(s) failed to download for '{s3_prefix}'.")
                else:
                    print(f"✅ Download complete for '{s3_prefix}'.")

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == 'AccessDenied':
                print(f"❌ Error: Access denied to bucket '{BUCKET_NAME}'. Check your AWS credentials.")
            else:
                print(f"❌ An AWS client error occurred: {e}")
            break  # Stop on critical errors
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            break

    print("-" * 50)
    print("🎉 All downloads finished!")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description=f"Download scene data from the S3 bucket '{BUCKET_NAME}'."
    )
    parser.add_argument(
        "scene_name",
        type=str,
        help="The name of the scene to download (e.g., 'berlin')."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel download workers (default: {DEFAULT_WORKERS})."
    )
    args = parser.parse_args()

    # Before running, make sure you have the necessary libraries
    try:
        from tqdm import tqdm as _
    except ImportError:
        print("Required package 'tqdm' not found. Please install it using: pip install tqdm")
        exit(1)

    download_scene_data(scene_name=args.scene_name, num_workers=args.workers)