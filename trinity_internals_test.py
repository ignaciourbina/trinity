"""
================================================================================
THE TRINITY: INTERNALS TESTS (PHASE 1)
================================================================================
CPU-feasible experiments to test whether GPT-2 internals align with Trinity
framework predictions.

Implements:
  1) Linear probes for V*(h_t)
  2) Advantage direction in representation space

Expected runtime: ~10-15 minutes on CPU.
================================================================================
"""

import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from transformers import GPT2LMHeadModel, GPT2Tokenizer

warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================================
# 1. LOAD MODEL
# ============================================================================

print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.eval()

EOS_ID = tokenizer.eos_token_id
VOCAB_SIZE = model.config.vocab_size
HIDDEN_DIM = model.config.n_embd
N_LAYERS_WITH_EMBED = model.config.n_layer + 1  # embedding + transformer layers

print(f"Vocab: {VOCAB_SIZE}, Hidden dim: {HIDDEN_DIM}, EOS token: {EOS_ID}")
print(f"Layers for probes: {N_LAYERS_WITH_EMBED} (embedding + {model.config.n_layer} blocks)")


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

PROMPTS = [
    "The meaning of life is",
    "In 1969, the first human",
    "The capital of France is",
    "Once upon a time there was a",
    "Quantum mechanics suggests that",
    "The recipe for perfect pasta starts with",
    "In a distant galaxy, explorers discovered",
    "The CEO announced that the company will",
    "To improve mental health, experts recommend",
    "The ancient philosophers argued that",
    "When programming in Python, it is important to",
    "Climate change policies should prioritize",
]

TOP_K = 10
DEPTH = 2
BETA = 1.0
RIDGE_ALPHA = 1.0
RNG_SEED = 42

np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)


# ============================================================================
# 3. REWARD + DP HELPERS (same structure as trinity_gpt2_colab.py)
# ============================================================================


def reward_function(token_ids, log_probs_under_ref):
    """Trajectory reward used for DP recursion."""
    r = 0.0
    for tid, lp in zip(token_ids, log_probs_under_ref):
        r += 0.5 * lp
        r -= 0.1 * max(lp + 3.0, 0)
    r += 0.2 * len(token_ids)
    if token_ids and token_ids[-1] == EOS_ID:
        r += 1.0
    return r


@torch.no_grad()
def forward_logits_hidden_all_layers(input_ids):
    """Return next-token logits and final-position hidden states for all layers."""
    input_tensor = torch.tensor([input_ids], device=device)
    outputs = model(input_tensor, output_hidden_states=True, use_cache=False)
    logits = outputs.logits[:, -1, :]
    hidden_all = torch.stack([h[0, -1, :] for h in outputs.hidden_states], dim=0)
    return logits, hidden_all


def get_top_k_with_eos(logits, k):
    """Get top-K candidates, ensuring EOS is included."""
    probs = F.softmax(logits, dim=-1)
    _, topk_ids = torch.topk(probs[0], k)
    topk_ids = topk_ids.detach().cpu().tolist()
    if EOS_ID not in topk_ids:
        topk_ids[-1] = EOS_ID
    return topk_ids


def compute_Z_recursive(input_ids, depth, beta, top_k,
                        token_history=None, logprob_history=None):
    """Backward DP for Z(h_t), mirroring the approach in trinity_gpt2_colab.py."""
    if token_history is None:
        token_history = []
    if logprob_history is None:
        logprob_history = []

    logits, _ = forward_logits_hidden_all_layers(input_ids)
    log_probs_ref = F.log_softmax(logits, dim=-1)
    probs_ref = torch.exp(log_probs_ref)
    candidate_tokens = get_top_k_with_eos(logits, top_k)

    if depth == 0:
        Z = 0.0
        per_token = {}
        for v in candidate_tokens:
            p_v = probs_ref[0, v].item()
            new_history = token_history + [v]
            new_logprobs = logprob_history + [log_probs_ref[0, v].item()]
            R_v = reward_function(new_history, new_logprobs)
            exp_r = float(np.exp(R_v / beta))
            Z += p_v * exp_r
            per_token[v] = {
                'p_ref': p_v,
                'reward': R_v,
                'exp_R_beta': exp_r,
                'is_eos': v == EOS_ID,
                'Z_next': exp_r,
                'reward_stop': R_v,
            }
        return float(Z), {
            'per_token': per_token,
            'candidate_tokens': candidate_tokens,
            'log_probs_ref': log_probs_ref.detach().cpu().numpy(),
        }

    Z = 0.0
    per_token = {}

    for v in candidate_tokens:
        p_v = probs_ref[0, v].item()
        new_history = token_history + [v]
        new_logprobs = logprob_history + [log_probs_ref[0, v].item()]

        if v == EOS_ID:
            R_stop = reward_function(new_history, new_logprobs)
            Z_next = float(np.exp(R_stop / beta))
            Z += p_v * Z_next
            per_token[v] = {
                'p_ref': p_v,
                'Z_next': Z_next,
                'is_eos': True,
                'reward_stop': R_stop,
            }
        else:
            next_input = input_ids + [v]
            Z_next, _ = compute_Z_recursive(
                next_input,
                depth - 1,
                beta,
                top_k,
                token_history=new_history,
                logprob_history=new_logprobs,
            )
            Z += p_v * Z_next
            per_token[v] = {
                'p_ref': p_v,
                'Z_next': Z_next,
                'is_eos': False,
            }

    return float(Z), {
        'per_token': per_token,
        'candidate_tokens': candidate_tokens,
        'log_probs_ref': log_probs_ref.detach().cpu().numpy(),
    }


# ============================================================================
# 4. EXPERIMENT 1: LINEAR PROBES FOR V*(h_t)
# ============================================================================


def collect_probe_dataset(prompts, beta=BETA, top_k=TOP_K, depth=DEPTH):
    """Collect (hidden_state, V*) pairs for every layer over prompt prefixes."""
    layer_features = [[] for _ in range(N_LAYERS_WITH_EMBED)]
    y_values = []

    print("\nCollecting probe dataset from prompt prefixes...")
    n_prefixes = 0

    for pidx, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) < 1:
            continue

        print(f"  [{pidx+1}/{len(prompts)}] {prompt[:50]}...")
        for t in range(1, len(token_ids) + 1):
            prefix = token_ids[:t]
            Z_t, _ = compute_Z_recursive(prefix, depth, beta, top_k)
            V_t = beta * np.log(max(Z_t, 1e-30))

            _, hidden_all = forward_logits_hidden_all_layers(prefix)
            hidden_np = hidden_all.detach().cpu().numpy()

            for layer_idx in range(N_LAYERS_WITH_EMBED):
                layer_features[layer_idx].append(hidden_np[layer_idx])
            y_values.append(V_t)
            n_prefixes += 1

    y = np.array(y_values, dtype=np.float64)
    X_by_layer = [np.array(layer_features[i], dtype=np.float32) for i in range(N_LAYERS_WITH_EMBED)]

    print(f"  Collected {n_prefixes} prefixes / {len(y)} labels")
    return X_by_layer, y


def run_linear_probes(X_by_layer, y, alpha=RIDGE_ALPHA):
    """Train ridge probes + permutation baseline for each layer."""
    r2_by_layer = []
    perm_r2_by_layer = []
    mlp_r2_by_layer = []

    best_layer = 0
    best_r2 = -1e9
    best_scatter = (np.array([]), np.array([]))

    if len(y) < 8:
        raise RuntimeError("Not enough samples for train/test split in probe experiment.")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=RNG_SEED)

    for layer_idx, X in enumerate(X_by_layer):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        probe = Ridge(alpha=alpha)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_by_layer.append(float(r2))

        y_train_perm = np.random.permutation(y_train)
        probe_perm = Ridge(alpha=alpha)
        probe_perm.fit(X_train, y_train_perm)
        y_pred_perm = probe_perm.predict(X_test)
        perm_r2 = r2_score(y_test, y_pred_perm)
        perm_r2_by_layer.append(float(perm_r2))

        mlp = MLPRegressor(
            hidden_layer_sizes=(64,),
            activation='relu',
            max_iter=300,
            random_state=RNG_SEED,
        )
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        mlp_r2 = r2_score(y_test, y_pred_mlp)
        mlp_r2_by_layer.append(float(mlp_r2))

        if r2 > best_r2:
            best_r2 = r2
            best_layer = layer_idx
            best_scatter = (y_test.copy(), y_pred.copy())

    return {
        'r2_by_layer': np.array(r2_by_layer),
        'perm_r2_by_layer': np.array(perm_r2_by_layer),
        'mlp_r2_by_layer': np.array(mlp_r2_by_layer),
        'best_layer': int(best_layer),
        'best_r2': float(best_r2),
        'best_scatter': best_scatter,
    }


# ============================================================================
# 5. EXPERIMENT 2: ADVANTAGE DIRECTION
# ============================================================================


def collect_advantage_pairs(prompts, beta=BETA, top_k=TOP_K, depth=DEPTH):
    """Collect pairwise (Δh, ΔA) data at final layer across prompts."""
    delta_h_all = []
    delta_a_all = []
    per_prompt_data = {}

    print("\nCollecting advantage-direction pairs...")
    for pidx, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt)
        Z_t, info = compute_Z_recursive(input_ids, depth, beta, top_k)
        V_t = beta * np.log(max(Z_t, 1e-30))

        candidates = info['candidate_tokens']
        vectors = []
        advantages = []

        for v in candidates:
            token_info = info['per_token'][v]
            if token_info.get('is_eos', v == EOS_ID):
                A_v = token_info['reward_stop'] - V_t
            else:
                Z_next = token_info['Z_next']
                V_next = beta * np.log(max(Z_next, 1e-30))
                A_v = V_next - V_t

            _, hidden_all = forward_logits_hidden_all_layers(input_ids + [v])
            h_next = hidden_all[-1].detach().cpu().numpy()

            vectors.append(h_next)
            advantages.append(A_v)

        prompt_delta_h = []
        prompt_delta_a = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                dA = advantages[i] - advantages[j]
                dH = vectors[i] - vectors[j]
                prompt_delta_h.append(dH)
                prompt_delta_a.append(dA)
                delta_h_all.append(dH)
                delta_a_all.append(dA)

        per_prompt_data[prompt] = {
            'delta_h': np.array(prompt_delta_h, dtype=np.float32),
            'delta_a': np.array(prompt_delta_a, dtype=np.float64),
        }
        print(f"  [{pidx+1}/{len(prompts)}] pairs: {len(prompt_delta_a)}")

    X = np.array(delta_h_all, dtype=np.float32)
    y = np.array(delta_a_all, dtype=np.float64)
    return X, y, per_prompt_data


def fit_advantage_direction(X, y, per_prompt_data, alpha=RIDGE_ALPHA):
    """Fit global direction u and compute prompt-wise consistency metrics."""
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    corr = float(np.corrcoef(y_pred, y)[0, 1]) if len(y) > 1 else 0.0
    u_global = reg.coef_.astype(np.float64)

    prompt_vectors = []
    prompt_names = []
    for prompt, data in per_prompt_data.items():
        Xp, yp = data['delta_h'], data['delta_a']
        if len(yp) < 2:
            continue
        reg_p = Ridge(alpha=alpha, fit_intercept=False)
        reg_p.fit(Xp, yp)
        u_p = reg_p.coef_.astype(np.float64)
        prompt_vectors.append(u_p)
        prompt_names.append(prompt)

    if not prompt_vectors:
        cos_matrix = np.zeros((0, 0))
    else:
        U = np.stack(prompt_vectors, axis=0)
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
        U = U / norms
        cos_matrix = U @ U.T

    return {
        'u_global': u_global,
        'y_true': y,
        'y_pred': y_pred,
        'corr': corr,
        'prompt_names': prompt_names,
        'cos_matrix': cos_matrix,
    }


# ============================================================================
# 6. FIGURES + REPORT
# ============================================================================


def make_figure(probe_res, adv_res, output_path='trinity_internals_results.png'):
    """Create summary figure with required 5 panels."""
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('The Trinity Internals Tests (GPT-2 124M, Phase 1)',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28,
                  left=0.06, right=0.97, top=0.94, bottom=0.05)

    # Panel 1: R² per layer + permutation baseline
    ax1 = fig.add_subplot(gs[0, 0])
    layers = np.arange(len(probe_res['r2_by_layer']))
    width = 0.35
    ax1.bar(layers - width / 2, probe_res['r2_by_layer'], width,
            label='Ridge probe', color='#2563eb', alpha=0.85)
    ax1.bar(layers + width / 2, probe_res['perm_r2_by_layer'], width,
            label='Permutation baseline', color='#94a3b8', alpha=0.9)
    ax1.plot(layers, probe_res['mlp_r2_by_layer'], 'o--',
             color='#ea580c', markersize=4, linewidth=1.2, label='MLP (64)')
    ax1.set_xlabel('Layer index (0 = embedding)')
    ax1.set_ylabel('R² on held-out split')
    ax1.set_title('Experiment 1: V* Linear Probes by Layer', fontweight='bold')
    ax1.set_xticks(layers)
    ax1.legend(fontsize=8)

    # Panel 2: best-layer scatter
    ax2 = fig.add_subplot(gs[0, 1])
    y_true, y_pred = probe_res['best_scatter']
    ax2.scatter(y_true, y_pred, alpha=0.7, s=35, c='#7c3aed')
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax2.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    ax2.set_xlabel('Actual V*')
    ax2.set_ylabel('Predicted V*')
    ax2.set_title(f'Best Layer {probe_res["best_layer"]} Scatter (R²={probe_res["best_r2"]:.3f})',
                  fontweight='bold')

    # Panel 3: predicted ΔA vs actual ΔA
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(adv_res['y_true'], adv_res['y_pred'], alpha=0.35, s=14, c='#dc2626')
    lo3 = min(np.min(adv_res['y_true']), np.min(adv_res['y_pred']))
    hi3 = max(np.max(adv_res['y_true']), np.max(adv_res['y_pred']))
    ax3.plot([lo3, hi3], [lo3, hi3], 'k--', linewidth=1)
    ax3.set_xlabel('Actual ΔA')
    ax3.set_ylabel('Predicted ΔA = u·Δh')
    ax3.set_title(f'Experiment 2: Advantage Direction (corr={adv_res["corr"]:.3f})',
                  fontweight='bold')

    # Panel 4: cosine similarity matrix
    ax4 = fig.add_subplot(gs[1, 1])
    cos_matrix = adv_res['cos_matrix']
    if cos_matrix.size == 0:
        ax4.text(0.5, 0.5, 'Not enough prompt-wise vectors', ha='center', va='center')
        ax4.set_axis_off()
    else:
        im = ax4.imshow(cos_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        ax4.set_title('Per-Prompt Advantage Direction Cosine Similarity', fontweight='bold')
        ticks = np.arange(len(adv_res['prompt_names']))
        ax4.set_xticks(ticks)
        ax4.set_yticks(ticks)
        ax4.set_xticklabels([str(i + 1) for i in ticks], fontsize=7)
        ax4.set_yticklabels([str(i + 1) for i in ticks], fontsize=7)
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # Panel 5: summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    mean_perm = float(np.mean(probe_res['perm_r2_by_layer']))
    best_mlp = float(np.max(probe_res['mlp_r2_by_layer']))

    if cos_matrix.size > 0:
        iu = np.triu_indices_from(cos_matrix, k=1)
        mean_cos = float(np.mean(cos_matrix[iu])) if len(iu[0]) else 1.0
    else:
        mean_cos = float('nan')

    summary_text = f"""
  SUMMARY (Phase 1 Internals Tests)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Setup:
    Prompts: {len(PROMPTS)}
    DP params: TOP_K={TOP_K}, DEPTH={DEPTH}, BETA={BETA}
    Layers tested: {N_LAYERS_WITH_EMBED}

  Experiment 1 (V* probes):
    Best ridge layer: {probe_res['best_layer']} (R²={probe_res['best_r2']:.4f})
    Mean permutation baseline R²: {mean_perm:.4f}
    Best MLP R² (optional nonlinear): {best_mlp:.4f}

  Experiment 2 (advantage direction):
    Corr(u·Δh, ΔA): {adv_res['corr']:.4f}
    Mean cosine similarity (prompt u vectors): {mean_cos:.4f}

  Outputs:
    Saved figure: trinity_internals_results.png
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    ax5.text(0.02, 0.98, summary_text, transform=ax5.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8fafc', alpha=0.9))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")


def print_summary(probe_res, adv_res):
    """Print clear text summary to stdout."""
    print("\n" + "=" * 80)
    print("TRINITY INTERNALS TEST SUMMARY")
    print("=" * 80)

    print("\nExperiment 1: Linear probes for V*(h_t)")
    for i, (r2, r2p) in enumerate(zip(probe_res['r2_by_layer'], probe_res['perm_r2_by_layer'])):
        print(f"  Layer {i:2d}: R²={r2:>8.4f}   perm={r2p:>8.4f}")
    print(f"\n  Best layer: {probe_res['best_layer']} (R²={probe_res['best_r2']:.4f})")
    print(f"  Best MLP R²: {np.max(probe_res['mlp_r2_by_layer']):.4f}")

    print("\nExperiment 2: Advantage direction")
    print(f"  Corr(predicted ΔA, actual ΔA): {adv_res['corr']:.4f}")

    cos_matrix = adv_res['cos_matrix']
    if cos_matrix.size > 0:
        iu = np.triu_indices_from(cos_matrix, k=1)
        mean_cos = np.mean(cos_matrix[iu]) if len(iu[0]) else 1.0
        print(f"  Mean cosine similarity of per-prompt u: {mean_cos:.4f}")
    else:
        print("  Mean cosine similarity of per-prompt u: N/A")

    print("\nOutput artifact:")
    print("  trinity_internals_results.png")


# ============================================================================
# 7. RUN PHASE 1
# ============================================================================


def main():
    t0 = time.time()

    print("\n" + "=" * 80)
    print("THE TRINITY: INTERNALS TESTS (PHASE 1)")
    print("=" * 80)
    print(f"Prompts: {len(PROMPTS)} | TOP_K={TOP_K} | DEPTH={DEPTH} | BETA={BETA}")

    X_by_layer, y = collect_probe_dataset(PROMPTS)
    probe_res = run_linear_probes(X_by_layer, y)

    X_adv, y_adv, per_prompt_data = collect_advantage_pairs(PROMPTS)
    adv_res = fit_advantage_direction(X_adv, y_adv, per_prompt_data)

    make_figure(probe_res, adv_res)
    print_summary(probe_res, adv_res)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")


if __name__ == '__main__':
    main()
