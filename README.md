# gLM2 BGC decoder

Conditional multi-domain protein sequence generation with the gLM2
biosynthetic gene cluster (BGC) diffusion decoder.

## Install

The script requires Python 3.10+ and a CUDA-capable GPU.

```bash
git clone https://github.com/TattaBio/gLM2_decoder.git
cd gLM2_decoder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Generate sequences

Run the included VL-Start PKS example:

```bash
python sample_glm2.py \
  --prompt-fasta examples/vl_start.fasta \
  --domains-file examples/vl_start_domains.txt \
  --mask-prob 0.7 \
  --num-samples 10 \
  --num-steps 100 \
  --output-path samples.fasta
```

Use your own protein sequence:

```bash
python sample_glm2.py \
  --prompt-fasta template.fasta \
  --mask-prob 0.5 \
  --num-samples 10 \
  --num-steps 100 \
  --output-path samples.fasta
```

Restrict redesign to a sequence range and optionally provide antiSMASH domain
annotations:

```bash
python sample_glm2.py \
  --prompt-fasta template.fasta \
  --maskable-ranges 133:995 \
  --domains AS-PKS_KS,AS-PKS_AT \
  --mask-prob 0.7 \
  --num-samples 10 \
  --num-steps 100
```

Ranges and protected indices are zero-based and inclusive. Sampling defaults to
temperature 0.7 and eta 0.1. By default, the number of diffusion steps equals
the number of masked tokens; use `--num-steps` to cap the number of model passes.

Pseudo-likelihood scoring is optional because it adds extra model passes. Add
`--score` to any command to score and rank the generated sequences.

## Citation

TODO

## License

This project is licensed under the [Apache License 2.0](LICENSE).
