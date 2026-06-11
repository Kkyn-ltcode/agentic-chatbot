python -m src.analysis.scalability_experiment --out_dir results/scalability_inference && \
python -m src.analysis.scalability_experiment --mode training --out_dir results/scalability_training && \
python -m src.analysis.scalability_experiment --device cpu --out_dir results/scalability_cpu
