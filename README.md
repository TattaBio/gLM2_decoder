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

```
@article {Lanclos2026.09.11.750945,
    author = {Lanclos, Nathan and Ibrahim, Kyrellos and Cornman, Andre and Huang, Marco and Gill, Vikram and Jain, Aalini and Abraham, Jonathan and Gin, Jennifer and Chen, Yan and Petzold, Christopher and Baerwald, Justin and Kortemme, Tanja and Keasling, Jay and Hwang, Yunha},
    title = {Generative Design of New-to-nature Biosynthetic Assembly Lines with Genomic Language Modeling},
    elocation-id = {2026.09.11.750945},
    year = {2026},
    doi = {10.64898/2026.09.11.750945},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2026/09/21/2026.09.11.750945},
    eprint = {https://www.biorxiv.org/content/early/2026/09/21/2026.09.11.750945.full.pdf},
    journal = {bioRxiv}
}
```

## License

Code and model weights are licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. Free for academic and research use.
