# SSTVM: Score-based Spatio-Temporal Variational Model

This repository provides the implementation of **SSTVM**, a modular framework for spatiotemporal forecasting. It integrates Variational Graph Autoencoders (VGAE) and Score-based Diffusion in latent space, enabling robust and efficient modeling of complex spatiotemporal dynamics under noisy or sparse data conditions.

---

## 📁 Supported Datasets

This project supports nine publicly available benchmark datasets:
- Los-loop
- PeMS08
- T-Drive
- Didi_chengdu
- AIR-BJ
- ECG5000
- Electricity
- Solar
- Exchange-Rate

Please create a `data/` folder under each dataset directory and place the corresponding dataset files inside it.

---

## 🔧 Environment Requirements

For `graph_generation`:
- `tensorflow`
- `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn`, `sklearn`
- `argparse`, `configparser`, `time`, `math`

For `prediction`:
- `torch`
- `tensorflow`
- `numpy`, `matplotlib`, `pandas`, `sklearn`
- `argparse`, `csv`, `time`, `shutil`

---

## 🚀 Run the Demo


Go to the corresponding dataset folder and run:

python main.py


---

## 🙏 Acknowledgment
We sincerely thank the original authors of DVGNN for their excellent open-source contribution, which served as the foundation of this repository.

We build upon their framework and extend it into the proposed SSTVM model, including new latent-space diffusion mechanisms, enhanced robustness, and broader dataset support.



```bash