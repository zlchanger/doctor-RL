base_model=$1
ckpt_path=$2
target_dir=$3

python verl/scripts/model_merger.py --backend fsdp --tie-word-embedding --hf_model_path $base_model --local_dir $ckpt_path --target_dir $target_dir
cp $base_model/vocab.json $target_dir
cp $base_model/tokenizer.json $target_dir
cp $base_model/tokenizer_config.json $target_dir