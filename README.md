# Building Data-Driven Occupation Taxonomies: A Bottom-Up Multi-Stage Approach via Semantic Clustering and Multi-Agent Collaboration

### under review


## Requirements

- python 3.10
- for arabic text cleaning, you need to install pyarabic and camel-tools (https://github.com/CAMeL-Lab/camel_tools)

```bash
conda create --name climb python=3.10 
conda activate climb
pip install -r requirements.txt
```

## Environment
- rename the .env.example to .env and fill in the API keys

## Usage

### Data
unzip the data.zip and put it in the data folder

### Palestine

run src/0.palestine.ipynb

### Botswana

run src/1.botswana.ipynb

### USA

run src/2.usa.ipynb


### Evaluation
run src/taxonomy_evaluation.py

### Train the model that classify the same occupation
- prepare the data
- run src/same_occupation_job_pair_sampling.py
- run src/same_occupation_classification.py