# 💊 AI Drug Solubility Predictor (QSAR Model)

## 📌 Overview
This project is a Bioinformatics & Cheminformatics application designed to assist in the **Drug Discovery** process. It uses **Machine Learning (Random Forest)** to predict the aqueous solubility (**LogS**) of chemical compounds based on their molecular structure.

Solubility is a critical physicochemical property in **Bioengineering**, determining drug bioavailability and efficacy.

## 🚀 Features
* **SMILES Input:** Accepts chemical structures in standard SMILES format.
* **Cheminformatics Engine:** Utilizes **RDKit** to calculate molecular descriptors (Molecular Weight, LogP, NumHDonors, etc.) based on **Lipinski's Rule of 5**.
* **AI Prediction:** Powered by a **Random Forest Regressor** trained on real-world experimental data.
* **Real-World Data:** The model is trained on the benchmark **Delaney (ESOL) Dataset**.

## 🛠️ Technologies Used
* **Python**
* **RDKit** (Cheminformatics)
* **Scikit-Learn** (Machine Learning)
* **Streamlit** (Web Interface)
* **Pandas & NumPy** (Data Processing)

## 🧬 How to Run locally
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
