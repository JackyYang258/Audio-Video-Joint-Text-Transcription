export PYTHONPATH=./:$PYTHONPATH
python -u main.py --config-dir configs/ \
  --config-name video2text.yaml \
  task.data=/data2/final_project_data/30h_data \
  task.label_dir=/data2/final_project_data/30h_data \
  task.tokenizer_bpe_model=/data2/final_project_data/spm1000/spm_unigram1000.model \
  model.pretrained_path=/data2/final_project_ckpt/pretrained_model.pth \
  hydra.run.dir=/home/ai_hw_18/group_18/results common.user_dir=`pwd`