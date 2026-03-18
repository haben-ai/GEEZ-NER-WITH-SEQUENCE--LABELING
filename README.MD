<h1>Geʽez Named Entity Recognition (NER)</h1>

<p align="center">
  <b>Low-resource NLP with Weak Supervision + Transformers</b><br/>
  Developed by: <b>Haben Eyasu</b>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/geez-ner.svg" />
  <img src="https://img.shields.io/pypi/pyversions/geez-ner.svg" />
  <img src="https://img.shields.io/badge/cython-build-blue.svg" />
  <img src="https://img.shields.io/github/actions/workflow/status/haben-ai/GEEZ-NER-WITH-SEQUENCE--LABELING/main.yml" />
  <img src="https://img.shields.io/github/license/haben-ai/GEEZ-NER-WITH-SEQUENCE--LABELING" />
  <img src="https://img.shields.io/github/downloads/haben-ai/GEEZ-NER-WITH-SEQUENCE--LABELING/total" />
  <img src="https://img.shields.io/badge/code%20style-PEP8-green.svg" />
</p>

<p align="center">
  <a href="#">GitHub Sponsors</a> • 
  <a href="#">Patreon</a> • 
  <a href="#">Open Collective</a>
</p>

---

<h2>📌 About</h2>

Geʽez Named Entity Recognition (NER) is a research-driven NLP project focused on solving one of the hardest problems in AI today: **building language technologies for low-resource languages**.

This project introduces a **practical and scalable pipeline** that leverages:

- 🌍 Cross-lingual transfer from English  
- 🧠 Weak supervision for automatic labeling  
- 🤖 Transformer-based deep learning using XLM-R  

The goal is to **bridge the data scarcity gap** and enable structured information extraction from Geʽez texts, which are historically rich but computationally underrepresented.

---

<h2>🚀 Project Overview</h2>

![THIS IS THE DATA](https://github.com/haben-ai/GEEZ-NER-WITH-SEQUENCE--LABELING/blob/main/nerbanner.png)

This project focuses on Named Entity Recognition (NER) for Ge'ez, an ancient Semitic language. Due to the limited availability of labeled datasets for Ge'ez, this implementation utilizes **Weak Supervision combined with XLM-RoBERTa (XLM-R)** to achieve robust results in a low-resource setting.

<b>Core Idea:</b> Transfer knowledge from English → Geʽez using aligned sentence pairs.

- **Approach:** Weak Supervision + XLM-R Fine-tuning  
- **Data Format:** Parallel JSON (Geʽez ↔ English)  
- **Goal:** Automatically label Geʽez tokens using English entities  

---


---

<h2>📊 Data Example</h2>

```json
{
  "gez": "ሔዋን ወቃየን ሖሩ እንተ ባቢሎን እምቅድመ ይፃኡ እምሰማርያ።",
  "eng": "Eve and Cain walked through Babylon before they came up from Samaria."
}
```
<h2>🛠️ Getting Started</h2>

<b>1. Install Dependencies</b>

pip install -r requirements.txt

<b>2. Run the Pipeline</b>
Open:

notebooks/ALL-IN-ONE.ipynb

This notebook walks through:

Data preprocessing

Weak supervision labeling

Model fine-tuning

Evaluation

<h2>📊 Methodology</h2>

Weak Supervision:
Use English entity annotations to generate pseudo-labels for Geʽez

Fine-tuning:
Adapt XLM-R to Geʽez script and semantics

Evaluation:
Validate performance using manually verified Geʽez entities

<h2>📸 Results Snapshot</h2>
![THIS IS THE DATA](https://github.com/haben-ai/GEEZ-NER-WITH-SEQUENCE--LABELING/blob/main/Screenshot%202026-03-02%20165905.png)
<h2>💡 Why This Matters</h2>

Enables NLP for historical and underrepresented languages

Reduces dependence on large labeled datasets

Provides a reusable pipeline for other low-resource languages

<h2>🤝 Support & Contributions</h2>

If you find this project valuable:

⭐ Star the repository

🍴 Fork and contribute

💬 Share feedback

<h2>📜 License</h2>

This project is licensed under the MIT License.

<h2>👨‍💻 Author</h2>

Haben Eyasu
AI Engineer | NLP Researcher | Low-resource Language Advocate
