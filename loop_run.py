import os
import yaml
import glob
import finetune_seq2seq

# Path to save loop
loop_path = 'config/seq2seq/loop'
# Args to changes to create loop
argvars = {
    'data_dir': [#'data/go/section_1', 'data/go/section_2', 'data/go/section_3', 'data/go/section_4', 
        'data/go/section_5'],
    'output_dir': [#'runs/section_1/test_GO_01', 'runs/section_2/test_GO_01', 'runs/section_3/test_GO_01', 'runs/section_4/test_GO_01', 
        'runs/section_5/test_GO_01']
    # 'data_dir': ['data/go/small', 'data/go/small'],
    # 'output_dir': ['runs/small_1', 'runs/small_2']
}

def create_loop_files(loop_path, argvars):
    # Load base args file
    with open(f'config/seq2seq/train.yaml', 'r') as f:
        user_args = yaml.load(f, Loader=yaml.FullLoader)

    # Create yaml files for each run
    os.makedirs(loop_path, exist_ok=True)
    for i in range(len(argvars['data_dir'])):
        tmp_args = user_args
        tmp_args.update(
            data_dir=argvars['data_dir'][i],
            output_dir=argvars['output_dir'][i],
        )
        with open(os.path.join(loop_path, f'loop_{i+1}.yaml'), 'w') as outfile:
            yaml.dump(tmp_args, outfile)

def main(loop_path, argvars):
    create_loop_files(loop_path, argvars)

    for args_file in glob.glob(loop_path+'/*'):
        finetune_seq2seq.main(args_file)

if __name__ == '__main__':
    main(loop_path, argvars)