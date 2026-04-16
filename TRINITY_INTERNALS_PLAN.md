# The Trinity: Testing Whether GPT-2's Internals Align with the Three Frameworks

## Motivation

The current verification script (`trinity_gpt2_colab.py`) confirms that the identity

$$\Gamma = \exp(A^*/\beta) = \pi^*/\pi_{\text{ref}} = d\kappa^*/d\kappa_{\text{ref}}$$

holds numerically — but this is an algebraic tautology. It follows from the definitions of $Z$, $\pi^*$, and $A^*$ and would hold for *any* reference policy $\pi_{\text{ref}}$, not just a transformer.

The deeper question is: **does GPT-2's internal representation naturally encode structures predicted by these three frameworks?** Since we have full access to hidden states, attention weights, and the residual stream, we can test this empirically.

---

## Experiment 1: Linear Probes for V*(h_t)

**Question:** Is the optimal value function linearly decodable from the transformer's hidden states?

**Method:**
1. For a corpus of prompts, compute $V^*(h_t) = \beta \log Z(h_t)$ externally via backward DP.
2. Extract GPT-2's hidden-state vectors $\mathbf{h}_t^{(\ell)}$ at each of the 12 layers.
3. Train a ridge regression (linear probe) per layer: $\hat{V}^* = \mathbf{w}^{(\ell)} \cdot \mathbf{h}_t^{(\ell)} + b$.
4. Report $R^2$ per layer.

**What a strong result looks like:**
- High $R^2$ (> 0.8) at one or more layers → the transformer's representations are naturally aligned with the optimal control framework.
- $R^2$ increasing across layers → the value function is progressively computed through the forward pass.
- Low $R^2$ everywhere → the value function is not linearly accessible, and the control framework, while algebraically valid, does not describe what the transformer is doing internally.

**Controls:**
- Shuffle labels (permutation test) to establish baseline $R^2$.
- Compare against nonlinear probe (small MLP) to check if the information is present but nonlinearly encoded.

---

## Experiment 2: Advantage Direction in Representation Space

**Question:** Does the geometry of GPT-2's hidden space encode the advantage function?

**Method:**
1. For each prompt and each candidate token $v$, compute $A^*(h_t, v)$.
2. Obtain the hidden state after appending $v$: $\mathbf{h}_{t+1}(v)$.
3. For each pair of tokens $(v_i, v_j)$, compute:
   - $\Delta A = A^*(h_t, v_i) - A^*(h_t, v_j)$
   - $\Delta \mathbf{h} = \mathbf{h}_{t+1}(v_i) - \mathbf{h}_{t+1}(v_j)$
4. Check whether there is a consistent direction $\mathbf{u}$ such that $\Delta A \approx \mathbf{u} \cdot \Delta \mathbf{h}$ across many prompts.

**Metrics:**
- Cosine similarity between the learned $\mathbf{u}$ vectors across different prompts (consistency).
- Correlation between $\mathbf{u} \cdot \Delta \mathbf{h}$ and $\Delta A$ (predictive power).

**What a strong result looks like:**
- A single direction in hidden space tracks the advantage function across prompts → the transformer's geometry encodes policy improvement signals.

---

## Experiment 3: Attention Patterns as Backward Messages

**Question:** Do attention heads naturally implement something analogous to the HMM backward recursion?

**Method:**
1. Compute $\Gamma(h_t, v)$ for each position and token.
2. Extract attention weights from all 12 layers × 12 heads.
3. For each head, measure the mutual information (or rank correlation) between its attention pattern and the $\Gamma$ values.
4. Identify heads whose attention most closely tracks the $\Gamma$ ratio.

**What a strong result looks like:**
- Specific attention heads show significant correlation with $\Gamma$ → the HMM backward-message structure is reflected in the attention mechanism.
- These heads are in later layers → the backward computation is performed late in the forward pass (consistent with how transformers build contextual representations).

**Caveats:**
- Attention weights are not directly comparable to $\Gamma$ without careful normalisation.
- Correlation does not prove the head is *computing* $\Gamma$; it may be computing something correlated with it.

---

## Experiment 4: Kernel Radon–Nikodym from Transition Dynamics

**Question:** Does the empirical hidden-state transition kernel reflect the measure change $d\kappa^*/d\kappa_{\text{ref}}$?

**Method:**
1. From a fixed state $h_t$, sample $N$ continuations under $\pi_{\text{ref}}$ (reference policy) → observe empirical distribution of $h_{t+1}$.
2. Sample $N$ continuations under $\pi^*$ (optimal policy, computed via DP) → observe empirical distribution of $h_{t+1}$.
3. Estimate the density ratio $d\kappa^*/d\kappa_{\text{ref}}$ in hidden space using a classifier-based density ratio estimator (train a logistic regression to distinguish $h_{t+1}$ samples from $\pi^*$ vs $\pi_{\text{ref}}$).
4. Compare estimated density ratio to the externally computed $\Gamma$.

**What a strong result looks like:**
- The classifier-estimated density ratio matches $\Gamma$ → the HMM measure-change framework describes the actual transition dynamics in representation space.

**Practical notes:**
- Requires many forward passes per prompt (sampling continuations). May need to restrict to a small number of prompts.
- Density ratio estimation in 768 dimensions is hard; consider projecting to a lower-dimensional subspace first (PCA or the advantage direction from Experiment 2).

---

## Experiment 5: Cross-Framework Representational Alignment (CKA)

**Question:** Which layers of GPT-2 are most aligned with which framework?

**Method:**
1. For each framework, define the "ideal" representation at time $t$:
   - **Control:** $V^*(h_t)$ (scalar) or the gradient $\nabla_h V^*$ (vector).
   - **Bayes:** The likelihood ratio vector $[\pi^*(v|h_t) / \pi_{\text{ref}}(v|h_t)]_{v \in \mathcal{V}}$.
   - **HMM:** The Gamma vector $[\Gamma(h_t, v)]_{v \in \mathcal{V}}$.
2. Compute Centered Kernel Alignment (CKA) between each layer's hidden representations and each framework's ideal representation across a corpus of prompts.
3. Plot CKA per layer per framework.

**What a strong result looks like:**
- Different layers peak for different frameworks → the transformer implements a pipeline where different stages correspond to different theoretical perspectives.
- All frameworks peak at the same layer → the three views converge in a single representation (consistent with the Trinity being a single object viewed three ways).

---

## Implementation Plan

### Phase 1 (CPU-feasible, ~10–15 min runtime)
- Experiment 1 (linear probes) — most tractable, clearest signal
- Experiment 2 (advantage direction) — geometric, interpretable

### Phase 2 (GPU recommended, longer runtime)
- Experiment 3 (attention ↔ Gamma)
- Experiment 5 (CKA)

### Phase 3 (GPU required, many samples)
- Experiment 4 (kernel density ratio)

### Script structure
- `trinity_internals_test.py` — implements Phase 1
- Generates diagnostic figures and a summary table of $R^2$ values per layer
- Designed to run on CPU in reasonable time

---

## Expected Outcomes and Interpretation

| Outcome | Interpretation |
|---|---|
| Linear probe $R^2$ > 0.8 at late layers | The transformer naturally computes something close to $V^*$ — the control framework is descriptive, not just prescriptive |
| Consistent advantage direction across prompts | The policy improvement signal has a fixed geometric signature in representation space |
| Attention heads correlate with $\Gamma$ | The HMM backward recursion is at least partially implemented by attention |
| Kernel density ratio ≈ $\Gamma$ | The three frameworks describe the same underlying computation |
| Nothing correlates | The Trinity is algebraically valid but has no privileged relationship to how transformers actually work — it's a useful *external* analysis tool, not a description of *internal* mechanisms |

---

## Why This Matters

If the internals align with the Trinity frameworks, it suggests that RLHF-tuned transformers are (approximately) solving the soft-optimal control problem described by the theory — not just fitting reward signal in some ad hoc way. This would:

1. Provide **mechanistic interpretability** for RLHF: the value function, advantage, and measure change would have concrete neural correlates.
2. Predict **failure modes**: if the linear probe $R^2$ degrades for certain prompt types, those are cases where the model's internal planning breaks down.
3. Suggest **better training objectives**: if the transformer already encodes $V^*$ linearly, one could add auxiliary losses to strengthen this, improving alignment stability.

If the internals *don't* align, that is equally informative — it means the Trinity is a useful theoretical lens but transformers achieve their performance through different computational strategies.