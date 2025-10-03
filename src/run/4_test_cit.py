import json
import os
import pathlib
from argparse import ArgumentParser
import pdb

import numpy as np
from random import sample
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import collate_fn
from ..data.datasets import polyvore
from ..evaluation.metrics import compute_cir_scores
from ..models.load import load_model
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str,
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str,
                        default=None)
    parser.add_argument('--seed', type=int,
                        default=42)
    parser.add_argument('--checkpoint', type=str,
                        default=None)
    parser.add_argument('--demo', action='store_true')

    return parser.parse_args()


def validation(args):
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)


    test = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='test',
        metadata=metadata,
        embedding_dict=embedding_dict
    )


    test_dataloader = DataLoader(
        dataset=test, batch_size=1, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn
    )


    item_dataset = polyvore.PolyvoreItemDataset(
        dataset_dir=args.polyvore_dir,
        metadata=metadata,
        embedding_dict=embedding_dict
    )


    all_items = sample(list(item_dataset), 500)

    import pdb; pdb.set_trace()

    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    model.eval()

    with torch.no_grad():
        all_item_embeddings = model(all_items, use_precomputed_embedding=True)  # (num_candidates, emb_dim)

    print(f"Candidate pool size: {len(all_items)}")

    pbar = tqdm(test_dataloader, desc=f'[Test] Triplet Dataset')

    all_preds, all_labels = [], []

    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break

        # Compute query embeddings
        batched_q_emb = model(data['query'], use_precomputed_embedding=True)  # (batch_sz, emb_dim)
        labels = torch.tensor([item_dataset.all_item_ids.index(data['answer'][j].item_id)
                               for j in range(len(data['answer']))]).cuda()

        #if batch_candidate_embs:
        if True:
            # Compute distances in batches to save memory
            batch_size = 512  # adjust if needed
            dists = []
            for start in range(0, all_item_embeddings.shape[0], batch_size):
                end = start + batch_size
                cand_emb_batch = all_item_embeddings[start:end].unsqueeze(0)  # (1, batch, emb_dim)
                q_exp = batched_q_emb.unsqueeze(1)  # (batch_sz, 1, emb_dim)
                dists_batch = torch.norm(q_exp - cand_emb_batch, dim=-1)  # (batch_sz, batch)
                dists.append(dists_batch)
            dists = torch.cat(dists, dim=1)  # (batch_sz, num_candidates)
        #else:
        #    q_exp = batched_q_emb.unsqueeze(1).repeat(1, all_item_embeddings.shape[0], 1)
        #    dists = torch.norm(q_exp - all_item_embeddings.unsqueeze(0), dim=-1)

        preds = torch.argmin(dists, dim=-1)  # closest candidate
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Optional: compute CIR score per batch
        score = compute_cir_scores(all_preds[-1], all_labels[-1])
        pbar.set_postfix(**score)

    all_preds = torch.cat(all_preds).cuda()
    all_labels = torch.cat(all_labels).cuda()
    final_score = compute_cir_scores(all_preds, all_labels)
    print(f"[Test] Triplet Dataset --> {final_score}")

    # Save results
    if args.checkpoint:
        result_dir = os.path.join(RESULT_DIR, args.checkpoint.split('/')[-2])
    else:
        result_dir = os.path.join(RESULT_DIR, 'triplet_demo')
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(final_score, f)
    print(f"[Test] Triplet Dataset --> Results saved to {result_dir}")



if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    validation(args)