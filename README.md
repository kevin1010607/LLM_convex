# LLM_convex

## Set up environment
```bash
# Load module
module purge
module load pkg/Anaconda3 cuda/11.7 compiler/gcc/11.2.0

# Create conda environment
conda create -n alpaca -y python=3.8 pip
conda activate alpaca

# Install required packages
git clone https://github.com/kevin1010607/LLM_convex.git
cd LLM_convex
pip install -U pip
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"

# Install huggingface and login
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token {your_token}
huggingface-cli whoami

# Backup the original transformer
export MODEL_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/transformers/models
mv $MODEL_PATH/llama $MODEL_PATH/llama_backup
```

## Use customized transformer

### Modify the parameter
- `llama/modeling_llama.py`: line 987-989
    - keep_layer: int
    - slope: float
    - threshold: float
- `llama/modeling_llama.py`: line 1073
    - lm_head: lm_head or None
- `llama/modeling_llama.py`: line 1032
    - use_cache: bool

### Update to your customized transformer
```bash
conda activate alpaca

export MODEL_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/transformers/models
cp -r llama $MODEL_PATH
```

## Simple generate
- Use `simple_generate.py` to run some example.
- Use `test.sh` to run on compute node.

```bash
# Run it directly
python simple_generate.py \
    --base_model meta-llama/Llama-2-7b-hf

# Run it on compute node
sbatch test.sh
```

## Run llm_eval
- Use `llm_eval/eval.sh` to run evaluation.

```bash
cd llm_eval
sbatch eval.sh
```