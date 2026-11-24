# 🏦 Bank Marketing — Case-Based Reasoning & RAG

An explainable **Case-Based Reasoning (CBR)** classifier combined with a **Retrieval-Augmented Generation (RAG)** pipeline: the system predicts whether a client will subscribe to a term deposit, then uses an LLM to **explain the prediction in natural language** from the most similar past clients.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20API-F55036?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-8957E5?style=flat)

> 👥 **Group project** — Université Paris Dauphine-PSL

> **My contribution:** the full **RAG pipeline** — embedding-based retrieval of similar clients, prompt augmentation, LLM generation via the Groq API, and its integration into the app.

---

## 🎯 Overview

| | |
|---|---|
| **Task** | Binary classification — will the client subscribe? (yes / no) |
| **Dataset** | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`) |
| **Challenge** | Strong class imbalance: ~88% "no" vs ~11% "yes" |
| **Metric** | Balanced accuracy |
| **Case base** | ~30,000 past clients |

The project has two parts:

### Part 1 — Experiments
- **Experiment 1:** comparison of several k-NN models with different decision rules (majority voting vs. balanced) against CBR (*Retrieve & Reuse*), evaluated with balanced accuracy.
- **Experiment 2:** comparison of four complete CBR cycles against k-NN.

### Part 2 — Prediction & explanation platform
- A CBR engine over a case base of ~30k clients that predicts new cases.
- Full CBR cycle: predictions can be **revised** and **retained**, with a reliability score attached to each retained case.
- A web interface that displays the results and their interpretations.

---

## 🧠 Architecture — how RAG works here

```
New client ──► [1] Retrieval ──► similar past clients
                                        │
                                        ▼
                               [2] Augmentation ──► structured prompt
                                                          │
                                                          ▼
                                                  [3] Generation (LLM)
                                                          │
                                                          ▼
                                       Natural-language explanation of the prediction
```

| Step | Module | Role |
|---|---|---|
| **1. Retrieval** | `src/embedding_retrieval.py` | Finds the most similar past clients using embeddings |
| **2. Augmentation** | `src/augmentation.py` | Formats the retrieved cases into a structured prompt (`SYSTEM_PROMPT` + `user_prompt`) |
| **3. Generation** | `src/rag_generator.py` | Sends the prompt to an LLM via the Groq API and returns an explanation |

The generation step is integrated into the app through a dedicated **"Generate explanations with AI"** button in the prediction tab.

---

## ⚙️ Setup

Python 3.9 or newer. From the project root (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the dataset at `data/raw/bank-additional-full.csv`.

### LLM API key (for RAG generation)

1. Get a free API key from [Groq](https://console.groq.com/).
2. Duplicate `.env.example`, rename it to `.env`, and add your key:

```
GROQ_API_KEY=gsk_your_api_key_here
```

> If no key is provided, the RAG module falls back to a **simulation mode**, so the app still runs.

---

## 🚀 Run the pipeline

```bash
# 1. Prepare the data
python src/preprocessing.py

# 2. Run the experiments and the retrieval (generates neighbor files and results)
python src/experiments.py --quick
python src/embedding_retrieval.py

# 3. Launch the server
python web/rag_server.py
```

Then open `web/index.html` (e.g. with VS Code *Live Server*).

**In the app:**
1. Go to the **Predict a client** tab and click **Retrieve similar clients and predict**.
2. Once the CBR prediction is displayed, click **Generate explanations with AI** to get the LLM explanation.

---

## 📁 Project structure

```
CBR-RAG/
├── src/
│   ├── preprocessing.py         # data preparation
│   ├── cbr.py                   # CBR logic (retrieve, reuse, revise, retain)
│   ├── baseline_knn.py          # k-NN baselines
│   ├── experiments.py           # evaluation & comparisons
│   ├── embedding_retrieval.py   # [1] retrieval with embeddings
│   ├── augmentation.py          # [2] prompt construction
│   ├── rag_generator.py         # [3] LLM generation (Groq API)
│   └── app.py                   # app integration
├── web/                         # web interface & RAG server
├── data/
│   ├── raw/                     # source dataset
│   └── processed/               # generated splits & model inputs
├── results/                     # experiment outputs
├── notebooks/
├── tests/
├── requirements.txt
└── .env.example
```
