.PHONY: all install data train quantize eval demo clean

PYTHON := python3
NOTEBOOK := pocket_agent_colab.ipynb
NOTEBOOK_EXEC := jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200

# Run full pipeline end-to-end via Jupyter (requires jupyter and nbconvert)
all: install
	$(NOTEBOOK_EXEC) --output $(NOTEBOOK) $(NOTEBOOK)
	@echo "Pipeline complete. Artifacts saved to ./artifacts/"

install:
	pip install -q \
		"transformers>=5.0,<6.0" \
		"accelerate>=0.27" \
		"peft>=0.10" \
		"trl>=0.8" \
		"bitsandbytes>=0.43" \
		"datasets>=2.18" \
		"sentencepiece" \
		"protobuf" \
		"gradio>=4.0" \
		"safetensors" \
		jupyter \
		nbconvert

# Run individual sections by executing only up to certain tagged cells.
# For local partial runs, open the notebook in Jupyter and run sections manually.
data:
	@echo "Run Sections 0-3 of $(NOTEBOOK) in Jupyter to generate training data."

train:
	@echo "Run Sections 4-5 of $(NOTEBOOK) in Jupyter to fine-tune the model."
	@echo "Requires a CUDA GPU (T4 or better)."

quantize:
	@echo "Run Section 7 of $(NOTEBOOK) in Jupyter to merge LoRA and quantize."

eval:
	$(PYTHON) -c "\
from starter.eval_harness_contract import run_evaluation; \
from inference import run; \
results = run_evaluation(run, 'starter/public_test.jsonl'); \
print('Results:', results)"

demo:
	$(PYTHON) -c "\
import gradio as gr; \
from inference import run; \
def chat(msg, history): \
    h = [{'role': r, 'content': c} for r, c in (history or [])]; \
    return run(msg, h); \
gr.ChatInterface(chat, title='Pocket-Agent Demo').launch()"

clean:
	rm -rf artifacts/__pycache__ .ipynb_checkpoints
	@echo "Clean complete (artifacts preserved)."
