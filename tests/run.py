import subprocess
import sys
import os

PATH_TO_TESTS = '/mnt/d/code/glue-factory/test_video'

def run_tests():
    """
    Run the test suite for the LightGlue model.
    """

    files = os.listdir(PATH_TO_TESTS)  # Ensure we are in the correct directory

    for i in range(len(files) - 1):
        try:
            # Run pytest on the test_my_lightglue.py file
            result = subprocess.run(
                [sys.executable, 'tests/test_my_lightglue.py', 
                '--checkpoint', 
                '/mnt/d/code/glue-factory/outputs/training/spherecraft_pretrain_lightglue_run3/checkpoint_best.tar',
                '--image0', 
                f'{PATH_TO_TESTS}/{files[i]}',
                '--image1',
                f'{PATH_TO_TESTS}/{files[i+1]}',
                '--output', 
                '/mnt/d/code/glue-factory/tests/result_3_fps'],
                check=True,
                capture_output=True,
                text=True
            )
            print("Tests passed successfully.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Tests failed.")
            print(e.stdout)
            print(e.stderr)

if __name__ == "__main__":
    run_tests()