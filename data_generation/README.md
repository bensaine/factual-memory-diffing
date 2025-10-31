# Data Generation Module

This module contains the synthetic data generation pipeline for factual memory diffing research.

## Quick Examples

```bash
cd data_generation

# Generate triplets
uv run --env-file .env -- python scripts/generate_triplets.py --num-triplets 100 --batch-size 35 --model gpt-4o --domains people-of-importance-politics people-of-importance-ceo people-of-importance-sports --after-year 2020 --output ../data/triplets.json 

# Verbalize with standard inference
uv run --env-file .env -- python scripts/verbalize_triplets.py --triplets ../data/triplets.json --inference standard --output ../data/verbalizations.json 

# Verbalize with creative inference
uv run --env-file .env -- python scripts/verbalize_triplets.py --triplets ../data/triplets.json --inference creative --output ../data/verbalizations.json 

# List available templates
python scripts/verbalize_triplets.py --list-templates
```

## Python API

```python
from data_generation import TripletGenerator, Verbalizer, InferenceTemplate

# Generate triplets
gen = TripletGenerator()
triplets = gen.generate_batch(num_triplets=100)

# Verbalize
verb = Verbalizer()
results = verb.verbalize_batch(
    triplets=[t.to_dict() for t in triplets],
    num_exposures=30
)
```