Geʽez Named Entity Recognition (NER)
Developed by: Haben Eyasu

Final Project Coursework | Natural Language Processing (NLP)

This project focuses on Named Entity Recognition (NER) for Ge'ez, an ancient Semitic language. Due to the limited availability of labeled datasets for Ge'ez, this implementation utilizes Weak Supervision combined with the XLM-RoBERTa (XLM-R) transformer model to achieve robust results in a low-resource setting.

🚀 Project Overview
The core of this project involves mapping entities from a high-resource language (English) to a low-resource language (Ge'ez) using parallel corpora.

Approach: Weak Supervision + XLM-R Fine-tuning.

Data Format: The model processes parallel JSON pairs where English sentences serve as the source for entity alignment.

Data Example
JSON
{
  "gez": "ሔዋን ወቃየን ሖሩ እንተ ባቢሎን እምቅድመ ይፃኡ እምሰማርያ።",
  "eng": "Eve and Cain walked through Babylon before they came up from Samaria."
}
📂 Repository Structure
For ease of use, the primary workflow is documented in a single, comprehensive notebook.

Root Directory
.
├── data/
│   ├── raw/             # Original parallel sentences
│   ├── weak_labels/     # Labels generated via weak supervision
│   └── processed/       # Final tensors/datasets for training
├── geez_ner_models/     # Saved model weights and configurations
│   ├── config.json
│   ├── tokenizer_config.json
│   └── tokenizer.json 
├── src/                 # Modular Python scripts
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── ALL-IN-ONE.ipynb # Main entry point: Preprocessing, Training, & Eval
├── requirements.txt     # Environment dependencies
└── README.md
🛠️ Getting Started
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Pipeline:
Open notebooks/ALL-IN-ONE.ipynb to step through the data preprocessing, model training, and performance evaluation.

📊 Methodology
Weak Supervision: Leveraging English entity markers to bootstrap labels for the Ge'ez tokens.

Fine-tuning: Adjusting the XLM-R base model to recognize Geez script patterns and semantic entities.

Evaluation: Tested against a subset of manually verified Geez entities to ensure accuracy.
Screenshot 2026-03-02 165905.png