# Cyber Threat AI

This repository implements a complete pipeline from federated edge pre-training and proactive threat simulation to multi-modal detection, uncertainty analytics, neuro-symbolic validation, RL decisioning, action orchestration, and FastAPI serving. [6][12][10][15]

## Quickstart
- Install: `poetry install` and ensure a GPU if available. [6]
- Train end-to-end: `poetry run python train.py --config configs/default.yaml`. [6]
- Serve: `poetry run uvicorn src.serve.api:app --host 0.0.0.0 --port 8080`. [6]
- Tests: `poetry run pytest -q`. [6]

## Notes
- ViT-style image encoder, GCN for graphs, and Transformer for text with cross-modal attention. [12][10][15]
- MC-Dropout for uncertainty, Integrated Gradients for explanations, neurosymbolic rules for validation. [23][24][35]
