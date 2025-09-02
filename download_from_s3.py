import os
import boto3
from botocore.exceptions import ClientError

# --- Configuration ---
BUCKET_NAME = "contineu-ai-dev-bucket"
S3_PREFIX = "67dbee940e56164c4d12d8e2/panoramic-views/67e56da1a54e4b0012d0f668/"
# S3_PREFIX = "67dbee940e56164c4d12d8e2/point-clouds/67e56da1a54e4b0012d0f668/"
LOCAL_DOWNLOAD_DIRECTORY = r"/mnt/d/code/glue-factory/data/finetuning/67dbee940e56164c4d12d8e2/67e56da1a54e4b0012d0f668/images"
# LOCAL_DOWNLOAD_DIRECTORY = r"/mnt/d/code/glue-factory/data/finetuning/67dbee940e56164c4d12d8e2/67e56da1a54e4b0012d0f668/pointcloud"

# --- Function to Download from S3 ---
def download_s3_folder(bucket_name, s3_prefix, local_dir):
    """
    Downloads all objects from a specified S3 prefix (folder) to a local directory,
    retaining the folder structure.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_prefix (str): The S3 prefix (folder path) to download from.
        local_dir (str): The local directory to save the files to.
    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')

    print(f"Starting download from s3://{bucket_name}/{s3_prefix} to {local_dir}")

    try:
        # Use a paginator to handle potentially large number of objects
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

        for page in pages:
            if "Contents" in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Calculate the relative path within the S3 prefix
                    # e.g., if s3_key is "folder/subfolder/file.txt" and s3_prefix is "folder/",
                    # relative_path will be "subfolder/file.txt"
                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    
                    # Construct the full local path for the file
                    local_file_path = os.path.join(local_dir, relative_path)

                    # Ensure the local directory exists before downloading
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    print(f"Downloading s3://{bucket_name}/{s3_key} to {local_file_path}")
                    s3_client.download_file(bucket_name, s3_key, local_file_path)
            else:
                print(f"No contents found in s3://{bucket_name}/{s3_prefix}")
        print("Download complete!")

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == 'NoSuchKey' or error_code == '404':
            print(f"Error: S3 prefix '{s3_prefix}' not found in bucket '{bucket_name}'.")
        elif error_code == 'AccessDenied':
            print(f"Error: Access denied to bucket '{bucket_name}' or prefix '{s3_prefix}'. Check your AWS credentials and permissions.")
        else:
            print(f"An AWS client error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    download_s3_folder(BUCKET_NAME, S3_PREFIX, LOCAL_DOWNLOAD_DIRECTORY)