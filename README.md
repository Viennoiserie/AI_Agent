---
title: Gaia Agent
emoji: 🚀
colorTo: indigo
colorFrom: yellow
hf_oauth: true
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## 🌐 Gaia Agent Overview

Gaia Agent is a LangGraph-powered tool-using assistant built for evaluation tasks. It leverages multiple retrieval and reasoning tools to answer questions using a multi-hop computation graph.

This space supports:

- Automatic question retrieval from a remote API.
- Graph-based reasoning via LangGraph.
- Tool-assisted question answering.
- Answer submission and evaluation.

## 📁 Project Structure

- `app.py`: Launches the Gradio interface and handles API interactions.
- `ai_agent.py`: Defines the assistant's tools and LangGraph logic.
- `PROMPT.txt`: Contains the base system prompt to guide answer formatting.
- `setup.ipynb`: Setup notebook for Supabase vector DB, prompt creation, and testing.
- `requirement.txt`: Dependencies list for this project.

## 🧠 Supported Tools

The assistant can use the following tools:
- Arithmetic tools: `add`, `subtract`, `multiply`, `divide`, `modulus`
- Search tools: `wiki_search`, `web_search`, `arvix_search`
- Vector DB Retriever: `similar_question_search`

## 🛠️ Setup Instructions

Use `setup.ipynb` to:

1. Install dependencies
2. Load and inspect sample JSONL data
3. Generate `PROMPT.txt`
4. Initialize a Supabase vector DB
5. Insert vectorized documents from Q&A metadata

## 🚀 Run the Agent

You can run and test your agent using the Gradio interface:

```bash
python app.py
```

### Instructions inside the app:
- Log into Hugging Face
- Click "Run Evaluation & Submit All Answers"
- View score and answers

## ✅ Answer Format

All answers from the agent will follow this strict format:

```
FINAL ANSWER: [your concise final answer]
```

Numbers should be plain (no commas or units unless specified). Strings should avoid articles and abbreviations.

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 More Info

For full configuration and development guide, refer to the Hugging Face Spaces Config Docs:
https://huggingface.co/docs/hub/spaces-config-reference
