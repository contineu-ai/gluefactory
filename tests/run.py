import subprocess
import sys
import os
from itertools import product

PATH1 = '/data/code/glue-factory/68bc279594638700124b9aa2'
PATH2 = '/data/code/glue-factory/68d5463994638700124b9ad2'

files1 = os.listdir(PATH1)  
files2 = os.listdir(PATH2)

pairs = product(files1, files2, repeat=1)

def run_tests():
    """
    Run the test suite for the LightGlue model.
    """

    for pair in pairs:
        try:
            # Run pytest on the test_my_lightglue.py file
            result = subprocess.run(
                [sys.executable, 'tests/test_my_lightglue.py', 
                '--checkpoint', 
                '/data'
                '/code/glue-factory/outputs/training/spherecraft_pretrain_lightglue_run3/checkpoint_best.tar',
                '--image0', 
                f'{PATH1}/{pair[0]}',
                '--image1',
                f'{PATH2}/{pair[1]}',
                '--output', 
                '/data/code/glue-factory/result_aa2_ad2'],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("Tests failed.")
            print(e.stdout)
            print(e.stderr)

if __name__ == "__main__":
    run_tests()