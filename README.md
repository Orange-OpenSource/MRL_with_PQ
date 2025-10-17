# DISCLAIMER

This is the repository related to the paper: L. Roque, Q. Lampin, L-A. Dufrène, G. Larue, G. Lefebvre, M. Assaad, "Compacting Semantic Matryoshka Representations with Product Quantization", accepted at the 2025 IEEE Globecom Workshops on AI native Communications Systems-towards Integrated Intelligent and Highly Efficient Communications Systems.

Some instructions about the usage of the current implementations:
- For publishing the paper at GLOBECOM 2025, we used the following resources from this repository:
  - "semantic-codec-main" (Folder)
  - "MTEB_files_to_replace" (Folder)
- The folder "MTEB_files_to_replace" contains the files to be replaced in the MTEB libray after installing its module. This is required to ensure that the MTEB classifier (a Logistic Regression as standard model) will perform its training phase over the training embeddings without any processing, and running a fully random undersampling. During the testing phase, the evaluated model ('StudyModel class') will take place and will be used by the trained classifier.
  - Examples of local paths to change the MTEB files:
    - "AbsTaskClassification.py": /opt/miniconda3/lib/python3.12/site-packages/mteb/abstasks/AbsTaskClassification.py
    - "ClassificationEvaluator.py": /opt/miniconda3/lib/python3.12/site-packages/mteb/evaluation/evaluators/ClassificationEvaluator.py
    - "Banking77Classification.py": /opt/miniconda3/lib/python3.12/site-packages/mteb/tasks/Classification/eng/Banking77Classification.py
- The modified version of "SST2BinarySentClassification.py" is located at "/semantic-processing/Codes/semantic-codec-main/mteb_local/SST2BinarySentClassification.py" and does not need to be added to the original MTEB module.
