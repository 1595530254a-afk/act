# Sources and third-party notices

- ACT policy: `nikodembartnik/act_rebot_cube`, revision `409b639946cc4140b3934a6e3d2c255f4335ee9d`, https://huggingface.co/nikodembartnik/act_rebot_cube . Publisher metadata identifies Apache-2.0. Original model files, processors and model card are retained unchanged.
- Observation fixtures: extracted from `nikodembartnik/rebot_cube`, revision `c5cdae89d2e1f474a8e2f1ac5f16302126d0c33f`, https://huggingface.co/datasets/nikodembartnik/rebot_cube . Publisher metadata identifies Apache-2.0. This bundle contains five selected observations rather than the full dataset. Decoded RGB pixel values are losslessly represented as uint8 arrays; states retain dataset values. `fixtures/provenance.json` records selected frames and conversion. The original dataset card is included in `sources/dataset_README.md`.
- Separate CPU and CUDA reference action arrays are outputs of the original pinned model on the same observations. They are numerical reproduction aids, not ground-truth actions or real-robot performance results. Different compute backends can produce different numerical outputs.
- uv 0.13.1 Windows x64 executable: https://github.com/astral-sh/uv . Distributed under MIT OR Apache-2.0; accompanying upstream license texts and software bill of materials are retained under `licenses/`.
- Python dependencies are not bundled as installed environments. Their pinned versions are in `requirements.lock.txt`; they are downloaded from their distribution sources and retain their respective licenses.

The packaging scripts, fixture extraction, installation paths and offline-check entrypoint were added for this project. No original model weights were changed. This bundle contains no robot-control implementation and no trained optimizer state.
