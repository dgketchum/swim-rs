import subprocess
import os


def list_and_copy_gcs_bucket(cmd_path, bucket_path, local_dir, glob='*', dry_run=False):
    """"""
    prepend = 'gs://'
    list_cmd = [cmd_path, 'ls', os.path.join(prepend, bucket_path)]
    list_process = subprocess.Popen(list_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = list_process.communicate()

    if stderr:
        raise ValueError(f'Error listing bucket: {stderr.decode()}')

    found_files = stdout.decode().strip().split('\n')
    files_to_copy = [f for f in found_files if glob in os.path.basename(f)]

    if dry_run:
        [print(f) for f in files_to_copy]
        return None

    for file_path in files_to_copy:
        if file_path:
            filename = os.path.basename(file_path)
            copy_cmd = [cmd_path, 'cp', file_path, os.path.join(local_dir, filename)]
            copy_process = subprocess.Popen(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, copy_stderr = copy_process.communicate()

            if copy_stderr:
                print(f'Copying {file_path}: {copy_stderr.decode()}')
            print(f'Copied {file_path} to {local_dir}')


if __name__ == '__main__':
    command = '/home/dgketchum/google-cloud-sdk/bin/gsutil'
    root = '/media/research/IrrigationGIS/swim'
    bucket = 'wudr'

    for mask in ['inv_irr', 'irr']:
        dst = os.path.join(root, 'examples/tutorial/landsat/extracts/etf/{}'.format(mask))
        glob_ = f'etf_{mask}'

        list_and_copy_gcs_bucket(command, bucket, dst, glob=glob_, dry_run=False)
# ========================= EOF ====================================================================
