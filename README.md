# ECS289A-CostAwareRecipeGenerator
We aim to build a cost-efficient recipe generator capable of recommending cheaper alternatives to existing dishes without compromising nutritional value or portion sizes. Rising food costs and growing interest in healthy, affordable eating highlight the need for intelligent recipe optimization tools.

# Dataset
Link: https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m

Extract the recipe_data.csv to the root directory of this repo

# Dev Setup / Run Instructions

Follow these steps to create a virtual environment, install dependencies, and run the embedding script.

```bash
# 1) Create and activate a venv (macOS / zsh)
python3 -m venv venv
source venv/bin/activate

# 2) Upgrade pip and install dependencies from requirements.txt
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the embedding script (from the repository root)
cd src
python3 generateEmbeddings.py --max-rows=1000
```

Notes:
- The embedding script expects an Ollama server (default: http://localhost:11434) to be running if you use the Ollama embedding function.
- Adjust `--max-rows` as needed for testing vs full dataset runs.
