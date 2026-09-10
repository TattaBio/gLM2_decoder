#!/usr/bin/env python3
"""Standalone single-GPU inference script for the gLM2 BGC decoder."""

import argparse
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "tattabio/gLM2_650M_bgc_decoder"


def read_fasta(path):
    """Read the first sequence in a FASTA file."""
    sequence = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    break
            elif line:
                sequence.append(line)
    return "".join(sequence)


def parse_ranges(value):
    if not value:
        return None
    return [tuple(map(int, item.split(":"))) for item in value.split(",")]


def parse_indices(value):
    if not value:
        return None
    return [int(item) for item in value.split(",")]


def mask_prompt(sequence, mask_prob, maskable_ranges=None, protected_indices=None):
    """Mask a sequence, optionally within selected zero-based inclusive ranges."""
    maskable = np.zeros(len(sequence), dtype=bool)
    if maskable_ranges:
        for start, end in maskable_ranges:
            maskable[start : end + 1] = True
    else:
        maskable[:] = True

    if protected_indices:
        maskable[protected_indices] = False

    candidates = np.flatnonzero(maskable)
    count = int(len(candidates) * mask_prob)
    selected = np.random.choice(candidates, size=count, replace=False)
    masked = np.zeros(len(sequence), dtype=bool)
    masked[selected] = True
    return "".join(
        "<mask>" if masked[index] else residue
        for index, residue in enumerate(sequence)
    )


def topk_lowest_masking(scores, cutoff_len):
    sorted_scores, _ = scores.sort(dim=-1)
    threshold = sorted_scores.gather(dim=-1, index=cutoff_len)
    return scores < threshold


def sample_from_categorical(logits, temperature=1.0):
    logits = logits.double()
    if temperature != 0.0:
        uniform = torch.rand_like(logits).clamp_(1e-8, 1 - 1e-8)
        gumbel = -torch.log(-torch.log(uniform))
        logits = logits / temperature + gumbel
    scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


@torch.inference_mode()
def path_planning_sampling(tokens, model, tokenizer, num_steps, temperature, eta):
    """Iteratively decode masked tokens with Path Planning (P2) sampling."""
    fixed = tokens != tokenizer.mask_token_id
    total_masked = (~fixed).sum(dim=1, keepdim=True)
    if total_masked.max().item() == 0:
        return tokens

    output = tokens.clone()
    prediction = output
    for step in range(1, num_steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(output).logits
        currently_masked = output == tokenizer.mask_token_id
        previously_unmasked = ~currently_masked & ~fixed
        prediction, scores = sample_from_categorical(logits, temperature)
        scores = scores.masked_fill(fixed, float("inf"))
        scores[previously_unmasked] *= eta

        fraction_remaining = 1.0 - step / num_steps
        number_to_mask = (total_masked.float() * fraction_remaining).long()
        keep_masked = topk_lowest_masking(scores, number_to_mask)
        output[keep_masked] = tokenizer.mask_token_id
        reveal = currently_masked & ~keep_masked
        output[reveal] = prediction[reveal]

    remaining = output == tokenizer.mask_token_id
    output[remaining] = prediction[remaining]
    return output


class DiffusionSampler:
    def __init__(self, model, tokenizer, temperature=0.7, eta=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.eta = eta

    def sample(self, prompt, num_steps=None):
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        masked_count = int((tokens == self.tokenizer.mask_token_id).sum())
        steps = masked_count if num_steps is None else min(masked_count, num_steps)
        output = path_planning_sampling(
            tokens, self.model, self.tokenizer, steps, self.temperature, self.eta
        )
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=False)[0]
        return "".join(decoded.split())


@torch.inference_mode()
def pseudo_likelihood(model, tokenizer, sequence, batch_size=128):
    """Compute masked pseudo-likelihood for a generated sequence."""
    input_ids = tokenizer(sequence, return_tensors="pt").input_ids[0]
    special_ids = set(tokenizer.all_special_ids)
    positions = [
        index
        for index, token in enumerate(input_ids.tolist())
        if token not in special_ids
    ]
    total_loss = 0.0

    for offset in tqdm(range(0, len(positions), batch_size), leave=False):
        batch_positions = positions[offset : offset + batch_size]
        batch = input_ids.repeat(len(batch_positions), 1)
        rows = torch.arange(len(batch_positions))
        columns = torch.tensor(batch_positions)
        labels = batch[rows, columns].cuda()
        batch[rows, columns] = tokenizer.mask_token_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch.cuda()).logits[rows.cuda(), columns.cuda()]
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()

    return np.exp(-total_loss / len(positions))


def save_fasta(sequences, scores, output_path):
    with open(output_path, "w") as handle:
        for index, sequence in enumerate(sequences, start=1):
            score = "" if scores is None else f"_score_{scores[index - 1]:.4g}"
            handle.write(f">sequence_{index}{score}\n{sequence}\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-fasta",
        required=True,
        help="FASTA file containing the template sequence",
    )
    domain_input = parser.add_mutually_exclusive_group()
    domain_input.add_argument(
        "--domains",
        help='Comma-separated domain tokens, e.g. "AS-PKS_KS,AS-PKS_AT,KR"',
    )
    domain_input.add_argument(
        "--domains-file",
        help="Text file containing one domain token per line",
    )
    parser.add_argument("--mask-prob", type=float, default=0.8)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Diffusion steps (default: number of masked tokens)",
    )
    parser.add_argument(
        "--maskable-ranges",
        help='Zero-based inclusive ranges, e.g. "133:995,1025:2005"',
    )
    parser.add_argument(
        "--protected-indices",
        help="Comma-separated zero-based positions that remain fixed",
    )
    parser.add_argument("--output-path", default="sampled_sequences.fasta")
    parser.add_argument(
        "--score",
        action="store_true",
        help="Compute pseudo-likelihood scores and rank the generated sequences",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable GPU.")

    torch.manual_seed(42)
    np.random.seed(42)
    template = read_fasta(args.prompt_fasta)
    domains = []
    if args.domains:
        domains = [domain.strip() for domain in args.domains.split(",")]
    elif args.domains_file:
        with open(args.domains_file) as handle:
            domains = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    domain_prompt = "".join(f"<{domain}>" for domain in domains)

    print(f"Loading {args.model_name} on cuda:0...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name, trust_remote_code=True, token=args.hf_token
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, token=args.hf_token
    )
    sampler = DiffusionSampler(model, tokenizer, args.temperature, args.eta)

    sequences, scores = [], []
    start_time = time.time()
    for index in range(args.num_samples):
        masked = mask_prompt(
            template,
            args.mask_prob,
            parse_ranges(args.maskable_ranges),
            parse_indices(args.protected_indices),
        )
        generated = sampler.sample(f"<+>{domain_prompt}{masked}", args.num_steps)
        sequence = re.sub(r"<[^>]+>", "", generated)
        sequences.append(sequence)
        if args.score:
            score = pseudo_likelihood(model, tokenizer, sequence)
            scores.append(score)
            print(f"[{index + 1}/{args.num_samples}] score={score:.4g}")
        else:
            print(f"[{index + 1}/{args.num_samples}]")

    if args.score:
        order = np.argsort(scores)[::-1]
        sequences = [sequences[index] for index in order]
        scores = [scores[index] for index in order]
    save_fasta(sequences, scores if args.score else None, args.output_path)
    print(f"Saved {len(sequences)} sequences to {args.output_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
