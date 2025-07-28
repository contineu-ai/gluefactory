import os
import boto3

LOCAL_DIRECTORY = r"/mnt/d/code/glue-factory/data/pretraining/pairs"
BUCKET_NAME = "lightglue-training-data"
S3_PREFIX = "pairs"

def upload_to_s3():
    s3_client = boto3.client('s3')

    for idx, filename in enumerate(os.listdir(LOCAL_DIRECTORY)):
        if idx < 22110:
            continue
        else:
            local_path = os.path.join(LOCAL_DIRECTORY, filename)

            if os.path.isfile(local_path):
                s3_key = os.path.join(S3_PREFIX, filename).replace("\\", "/")
                print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}")
                s3_client.upload_file(local_path, BUCKET_NAME, s3_key)

if __name__ == "__main__":
    upload_to_s3() 