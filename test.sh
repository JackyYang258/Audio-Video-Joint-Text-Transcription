python -B inference.py --config-dir ./configs/ --config-name inference.yaml \
  dataset.gen_subset=test \
  common_eval.path="abstract path of your checkpoint" \
  common_eval.results_path="abstract path to the folder where you want to save results" \
  override.modalities="['video'] or ['video','audio'] based on which model to test" \
  common.user_dir=`pwd`