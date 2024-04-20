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
```

## Simple generate
- Use simple_generate.py to run some example.
- Use test.sh to run on compute node.

```bash
# Run it directly
python simple_generate.py \
    --base_model meta-llama/Llama-2-7b-hf

# Run it on compute node
sbatch test.sh
```
