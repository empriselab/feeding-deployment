 # LinUCB Variants for Mealtime Preference Prediction

## References and Inspiration

The contextual bandit formulation for mealtime preference prediction draws on three key papers:

1. **LinUCB algorithm.** Li, Chu, Langford, and Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation," WWW 2010 (arXiv:1003.0146). Introduced the LinUCB algorithm with disjoint and hybrid linear models for online personalization with contextual features. Our vanilla LinUCB baseline (Section 1) directly implements Algorithm 1 from this paper, adapted from article recommendation to per-dimension preference prediction.

2. **LinUCB for robot-assisted feeding with warm-start.** Gordon, Meng, Bhattacharjee, Barnes, and Srinivasa, "Adaptive Robot-Assisted Feeding: An Online Learning Framework for Acquiring Previously Unseen Food Items," ISRR 2019 (arXiv:1908.07088). Applied LinUCB to select bite acquisition strategies (fork pitch $\times$ fork roll) for previously unseen food items. Their key design choice was using a pre-trained neural network (SPANet) as a frozen featurizer, with only the final linear layer updated online. Their Experiment 2 demonstrated the warm-start paradigm: $\boldsymbol{\theta}$ was pre-trained on $\sim$8000 attempts on previously seen food items, then LinUCB adapted online to unseen foods. Our LLM-warm variant (Section 3) follows the same paradigm but replaces SPANet pre-training with LLM-generated pseudo-rewards, and our LLM-embed variant (Section 2) replaces SPANet's penultimate layer features with LLM embeddings.

3. **Contextual bandit for reflection selection in LLM-based recommendation.** Qin, Xu, Yu et al., "MoRE: A Mixture of Reflectors Framework for Large Language Model-Based Sequential Recommendation," RecSys 2025. Used a contextual bandit (PPO-based, 3-arm) to dynamically select among reflection perspectives for each user. While MoRE operates in a different domain (sequential item recommendation), it validates the use of bandit mechanisms for online selection in LLM-augmented systems and motivates our exploration of how much of the LLM's value comes from the prior knowledge versus inference-time reasoning.

The critical difference between Gordon et al.'s setting and ours is dimensionality: they predict a **single** discrete action ($K = 6$ strategies), while we predict a **$D$-dimensional bundle** ($D = 19$ preference dimensions). This forces the decomposition into $D$ independent bandits, which introduces the structural limitation of dimension independence — the central gap our full system addresses.

---

## 1. Vanilla LinUCB (Baseline)

### Problem Decomposition

We decompose the $D$-dimensional preference prediction problem into $D$ independent contextual bandit instances. For each dimension $d \in \{1, \ldots, D\}$ with $K_d$ possible values, we instantiate a $K_d$-armed LinUCB.

### Context Encoding

At meal episode $t$, the context $X_t = (\text{meal}, \text{setting}, \text{time\_of\_day}, U^{\text{phys}})$ is encoded via one-hot concatenation:

$$\mathbf{x}_t = \phi_{\text{onehot}}(X_t) = [\text{onehot}(\text{meal}) \,|\, \text{onehot}(\text{setting}) \,|\, \text{onehot}(\text{time\_of\_day}) \,|\, \text{onehot}(U^{\text{phys}}) \,|\, 1] \in \mathbb{R}^p$$

### Modeling Assumption

For each dimension $d$ and arm $a$, the expected reward is linear in the context:

$$\mathbb{E}[r_{t,a}^{(d)} | \mathbf{x}_t] = \mathbf{x}_t^\top \boldsymbol{\theta}_a^{(d)*}$$

### Initialization

$$\mathbf{A}_a^{(d)} \leftarrow \mathbf{I}_p, \qquad \mathbf{b}_a^{(d)} \leftarrow \mathbf{0}_{p \times 1}$$

### Arm Selection

$$a_t^{(d)} = \arg\max_{a \in \{1, \ldots, K_d\}} \left( \mathbf{x}_t^\top \hat{\boldsymbol{\theta}}_a^{(d)} + \alpha \sqrt{\mathbf{x}_t^\top (\mathbf{A}_a^{(d)})^{-1} \mathbf{x}_t} \right)$$

where $\hat{\boldsymbol{\theta}}_a^{(d)} = (\mathbf{A}_a^{(d)})^{-1} \mathbf{b}_a^{(d)}$.

### Online Update (Semi-Bandit Feedback)

After meal $t$, for each dimension $d$:

**If not corrected** (prediction accepted, $r = 1$):

$$\mathbf{A}_{a_t^{(d)}}^{(d)} \leftarrow \mathbf{A}_{a_t^{(d)}}^{(d)} + \mathbf{x}_t \mathbf{x}_t^\top, \qquad \mathbf{b}_{a_t^{(d)}}^{(d)} \leftarrow \mathbf{b}_{a_t^{(d)}}^{(d)} + 1 \cdot \mathbf{x}_t$$

**If corrected** to value $p_t^{*(d)}$ (two updates — semi-bandit):

$$\mathbf{A}_{a_t^{(d)}}^{(d)} \leftarrow \mathbf{A}_{a_t^{(d)}}^{(d)} + \mathbf{x}_t \mathbf{x}_t^\top, \qquad \mathbf{b}_{a_t^{(d)}}^{(d)} \leftarrow \mathbf{b}_{a_t^{(d)}}^{(d)} + 0 \cdot \mathbf{x}_t$$

$$\mathbf{A}_{p_t^{*(d)}}^{(d)} \leftarrow \mathbf{A}_{p_t^{*(d)}}^{(d)} + \mathbf{x}_t \mathbf{x}_t^\top, \qquad \mathbf{b}_{p_t^{*(d)}}^{(d)} \leftarrow \mathbf{b}_{p_t^{*(d)}}^{(d)} + 1 \cdot \mathbf{x}_t$$

### Correction Burden

$$C_t = \sum_{d=1}^{D} \mathbb{1}\left[a_t^{(d)} \neq p_t^{*(d)}\right]$$

### Structural Limitations

- Dimensions are independent: $p_t^{(d)} \perp p_t^{(d')} | \mathbf{x}_t$
- No bundle correlation modeling
- No sequential memory (stationary $\boldsymbol{\theta}_a^{(d)*}$)
- No mid-meal correction propagation across dimensions

---

## 2. LinUCB with LLM Embeddings

### Motivation

Replace hand-crafted one-hot encoding with semantic embeddings so that similar contexts (e.g., "pasta at dinner" and "noodles at dinner") share information via proximity in embedding space.

### Context Encoding

$$\mathbf{x}_t = \phi_{\text{LLM}}(X_t) \in \mathbb{R}^q$$

where $\phi_{\text{LLM}}$ is a frozen embedding model (e.g., OpenAI `text-embedding-3-small`, $q = 1536$). The input string is:

$$X_t = \text{"User: } U^{\text{phys}} \text{. Meal: } m_t \text{. Setting: } s_t \text{. Time: } \tau_t\text{"}$$

### Everything Else

Initialization, arm selection, and update rules are identical to vanilla LinUCB, but with $\mathbf{A}_a^{(d)} \in \mathbb{R}^{q \times q}$ and $\mathbf{b}_a^{(d)} \in \mathbb{R}^q$.

### Computational Note

The matrix inverse cost increases from $O(p^3)$ to $O(q^3)$ per step. In practice, this is still negligible compared to an LLM generative call. The inverse can also be computed periodically rather than per-step.

### LLM Cost

One embedding call per meal (fast, non-generative).

### What This Tests

Whether implicit semantic similarity in context space improves generalization, without any explicit world knowledge about preferences.

---

## 3. LinUCB with LLM-Warm Prior

### Motivation

Give LinUCB a "warm start" by querying a generative LLM offline to produce pseudo-rewards that initialize $\mathbf{A}_a^{(d)}$ and $\mathbf{b}_a^{(d)}$ with informed priors, so the system doesn't start from zero.

### Pseudo-Reward Generation (Offline, One-Time)

Let $\mathcal{X} = \{X_1, \ldots, X_N\}$ denote the set of anticipated mealtime contexts, constructed from the Cartesian product of meals, settings, times of day, and physical profiles. For each dimension $d$, arm $a$, and context $X_j$:

$$\tilde{r}_j^{(d,a)} = \text{LLM}_{\text{gen}}\left(\text{"Given context } X_j\text{, how likely does the user prefer value } a \text{ for dimension } d\text{?"}\right) \in [0, 1]$$

### Initialization with Prior

Let $\mathbf{x}_j = \phi_{\text{onehot}}(X_j)$. The sufficient statistics are initialized with pseudo-observations weighted by a trust parameter $\lambda > 0$:

$$\mathbf{A}_a^{(d)} \leftarrow \mathbf{I}_p + \lambda \sum_{j=1}^{N} \mathbf{x}_j \mathbf{x}_j^\top$$

$$\mathbf{b}_a^{(d)} \leftarrow \lambda \sum_{j=1}^{N} \tilde{r}_j^{(d,a)} \, \mathbf{x}_j$$

The initial weight estimate is:

$$\hat{\boldsymbol{\theta}}_{a,0}^{(d)} = \left(\mathbf{I}_p + \lambda \sum_{j=1}^{N} \mathbf{x}_j \mathbf{x}_j^\top\right)^{-1} \left(\lambda \sum_{j=1}^{N} \tilde{r}_j^{(d,a)} \, \mathbf{x}_j\right)$$

### Trust Parameter $\lambda$

- $\lambda = 0$: reduces to vanilla LinUCB (no prior)
- $\lambda \to \infty$: real corrections are overwhelmed by the prior; model becomes a static LLM prediction
- After $T$ real meals, the effective weight of the prior relative to real data scales as $\frac{\lambda N}{\lambda N + T}$, decaying naturally

### Online Update

Identical to vanilla LinUCB. The prior is progressively diluted as real observations accumulate.

### LLM Cost

$$\text{Total calls} = \sum_{d=1}^{D} K_d \times N$$

For $D = 19$ dimensions, $\sim 3$ values each, $N = 180$ contexts: $\approx 10{,}000$ calls. Incurred once before deployment.

### What This Tests

Whether explicit world knowledge injected as a prior improves cold-start performance, even when the reasoning capacity is not available at inference time.

---

## 4. LinUCB with LLM-Warm Prior + LLM Embeddings (Combined)

### Motivation

Combine both enhancements to create the strongest possible LinUCB variant. This has both explicit world knowledge (via warm-start) and implicit semantic generalization (via embeddings).

### Pseudo-Reward Generation

Same as Section 3, but contexts are embedded rather than one-hot encoded:

$$\tilde{r}_j^{(d,a)} = \text{LLM}_{\text{gen}}\left(\text{query about } X_j, d, a\right), \qquad \mathbf{x}_j = \phi_{\text{LLM}_{\text{embed}}}(X_j) \in \mathbb{R}^q$$

### Initialization

$$\mathbf{A}_a^{(d)} \leftarrow \mathbf{I}_q + \lambda \sum_{j=1}^{N} \mathbf{x}_j \mathbf{x}_j^\top \in \mathbb{R}^{q \times q}$$

$$\mathbf{b}_a^{(d)} \leftarrow \lambda \sum_{j=1}^{N} \tilde{r}_j^{(d,a)} \, \mathbf{x}_j \in \mathbb{R}^{q}$$

### Online Phase

Identical to vanilla LinUCB with $\mathbf{x}_t = \phi_{\text{LLM}_{\text{embed}}}(X_t)$.

### LLM Cost

- Init: $\sum_{d=1}^{D} K_d \times N$ generative calls (one-time)
- Per meal: 1 embedding call (cheap, non-generative)

### What This Tests

If the full LLM-based system (Ours) still outperforms this variant, the gap is attributable to **inference-time reasoning** — the capacity to interpret corrections in context, compose episodic evidence with world knowledge, and reason about bundle correlations dynamically — rather than mere access to world knowledge.

---

## Summary Table

| Variant | Prior $\hat{\boldsymbol{\theta}}_0$ | Features $\mathbf{x}_t$ | LLM calls at init | LLM calls per meal | Captures bundle correlations |
|---|---|---|---|---|---|
| Vanilla | $\mathbf{0}$ | One-hot $\in \mathbb{R}^p$ | 0 | 0 | No |
| LLM-embed | $\mathbf{0}$ | $\phi_{\text{LLM}} \in \mathbb{R}^q$ | 0 | 1 (embed) | No |
| LLM-warm | Informed | One-hot $\in \mathbb{R}^p$ | $\sum_d K_d \times N$ | 0 | No |
| LLM-both | Informed | $\phi_{\text{LLM}} \in \mathbb{R}^q$ | $\sum_d K_d \times N$ | 1 (embed) | No |
| **Ours** | N/A | Natural language | 0 | 1 (full gen) | **Yes** |

The rightmost column is the structural limitation shared by **all** LinUCB variants regardless of how much LLM knowledge they absorb: dimension independence $p_t^{(d)} \perp p_t^{(d')} | \mathbf{x}_t$ is baked into the architecture.

---

## Key Argument

The progression from vanilla → LLM-embed → LLM-warm → LLM-both systematically controls for two factors:

1. **Access to world knowledge** (warm-start provides explicit priors)
2. **Representational capacity** (embeddings provide semantic generalization)

If performance improves along this progression but still falls short of the full system, the remaining gap is due to **architectural capabilities** that no amount of prior knowledge can provide within the LinUCB framework:

- Bundle correlation modeling (cross-dimension reasoning)
- Inference-time compositional reasoning (interpreting corrections in context)
- Episodic memory (retrieving relevant past experiences)
- Non-stationary adaptation (tracking preference drift without re-training)