import os
import yaml
import glob
import finetune_seq2seq
import subprocess

# Path to save loop
loop_path = 'config/seq2seq/loop'
# Args to changes to create loop
p, q = 4, 6
argvars = {
    'data_dir': [f'data/go/section_{i}' for i in range(p, q)],
    'output_dir': [f'runs/autoCW/section_{i}/test_GO_01' for i in range(p, q)],
}

def create_loop_files(loop_path, argvars):
    # Load base args file
    with open(f'config/seq2seq/loop_base.yaml', 'r') as f:
        user_args = yaml.load(f, Loader=yaml.FullLoader)

    # Create yaml files for each run
    os.system('rm -rf ' + loop_path)
    os.makedirs(loop_path, exist_ok=True)
    for i in range(len(argvars['data_dir'])):
        tmp_args = user_args
        tmp_args.update(
            data_dir=argvars['data_dir'][i],
            output_dir=argvars['output_dir'][i],
        )
        n = argvars['data_dir'][i][-1]
        with open(os.path.join(loop_path, f'loop_{n}.yaml'), 'w') as outfile:
            yaml.dump(tmp_args, outfile)

def main(loop_path, argvars):
    create_loop_files(loop_path, argvars)

    for args_file in glob.glob(loop_path+'/*'):
    #     finetune_seq2seq.main(args_file)
        print('\nRunning: \n\tpython finetune_seq2seq.py '+ os.path.abspath(args_file))
        subprocess.run(['python finetune_seq2seq.py ' + os.path.abspath(args_file)], shell=True, check=True)


if __name__ == '__main__':
    main(loop_path, argvars)