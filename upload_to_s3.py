import os
import boto3
import tqdm

LOCAL_DIRECTORY = r"/data/Code/glue-factory/datasets/spherecraft_data/bank/gt_correspondences"
BUCKET_NAME = "spherecraft-dataset"
S3_PREFIX = "bank/gt_correspondences"

def upload_to_s3():
    s3_client = boto3.client('s3')

    for root, dirs, files in os.walk(LOCAL_DIRECTORY):
        for filename in tqdm.tqdm(files):
            local_path = os.path.join(root, filename)

            # Determine the S3 key, preserving the folder structure
            # We use os.path.relpath to get the path relative to the base directory
            # and then join it with the S3 prefix.
            relative_path = os.path.relpath(local_path, LOCAL_DIRECTORY)
            s3_key = os.path.join(S3_PREFIX, relative_path).replace("\\", "/")

            # print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}")
            s3_client.upload_file(local_path, BUCKET_NAME, s3_key)

if __name__ == "__main__":
    upload_to_s3()
    