import os
import sys
import shutil
import pandas as pd

BASEPATH = 'resource/'
WORKING = 'working/'

CSVPATH = 'test_v2_file_mapping.csv'
TIFPATH = os.path.join(BASEPATH, 'test-tif-v2')

FIXEDPATH = os.path.join(WORKING, 'fixed')

df = pd.read_csv(CSVPATH)


def fix_submission():
    # loading the data
    df_train = pd.read_csv('submission_0.csv')

    df_test = pd.DataFrame([[p.replace('.jpg', ''), t] for p, t in df_train.values])
    df_test.columns = ['image_name', 'tags']
    df_test.to_csv('submission_0_fixed.csv', index=False)


def copy_and_rename():
    '''Copy up to `num_files` images to the scratch directory.
    `num_files` is needed because you can only write a few hundred
    megabytes in this kernel environment. Use the `df -h` command
    to check.

    This is a purposely non-destructive operation. You'll need to
    move the renamed files back to the test-tif-v2 directory so
    that your existing scripts will continue to work.
    '''

    if not os.path.exists(FIXEDPATH):
        os.mkdir(FIXEDPATH)

    for index, row in df.iterrows():
        old = os.path.join(TIFPATH, row['old'])
        new = os.path.join(FIXEDPATH, row['new'])

        # copy file
        shutil.copy(old, new)

        # remove old
        os.remove(old)

        print '{} copied'.format(index)


if __name__ == '__main__':
    fix_submission()
