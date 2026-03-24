# Fine-Tuned Sentence-BERT for HPC Job Outcome Prediction via Textual Feature Embedding

This repository contains the source code for the paper **Fine-Tuned Sentence-BERT for HPC Job Outcome Prediction via Textual Feature Embedding**, published at HPCAsia 2026.

## Repository Structure

- `docs/`: Documentation describing the dataset features and dataset summaries (`.json`, `.csv`, `.md`).
- `generation_scripts/`: Python and shell scripts used for data preprocessing, anonymization, and feature generation pipeline.
- `src/`: Source code for running experiments, including baseline methods (`baseline_experiments.py`), fine-tuning SBERT (`finetune_SBERT_multitask.py`), and semantic analysis (`semantic_experience.py`).
- `notebooks/`: Jupyter notebooks (`EDA.ipynb`, `plot_results.ipynb`) for Exploratory Data Analysis and results visualization.
- `baseline_results/` & `predict_results/`: Output files containing evaluation metrics and predictions from various models.
- `plots/` & `F-Data-charts_pdf/`: Generated plots, distributions, and charts analyzing the dataset and model evaluation.
- `requirements.txt`: Python package dependencies required to run the code.
- `TUTORIALS.md`: Further instructions or guides on how to reproduce the results.

## Citation

If you use this code in your research, please cite:

```
@inproceedings{hai2026fine,
  title={Fine-Tuned Sentence-BERT for HPC Job Outcome Prediction via Textual Feature Embedding},
  author={Hai, Thanh Hoang Le and Tuan, Huy Nguyen and Dang, Bao Tran and Thuong, Bao Vo and Thoai, Nam},
  booktitle={Proceedings of the Supercomputing Asia and International Conference on High Performance Computing in Asia Pacific Region Workshops},
  pages={378--387},
  year={2026}
}
```
