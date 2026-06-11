## List of Symbols

| Symbol | Meaning |
|---|---|
| $s_t$ | State at time step $t$ |
| $a_t$ | Action taken at time step $t$ |
| $r_t$ | Reward received at time step $t$ |
| $s_{t+1}$ | Next state after taking action $a_t$ in state $s_t$ |
| $\pi$ | A policy — a mapping from states to actions |
| $\pi_\beta$ | Behavioral policy — the policy that collected dataset $\mathcal{D}$ |
| $\pi_{\text{off}}$ | Learned offline policy — the policy trained on $\mathcal{D}$ |
| $\pi_k$ | Policy at iteration $k$ during online/off-policy training |
| $\mathcal{D}$ | The offline dataset of transitions $(s_t, a_t, s_{t+1}, r_t)$ |
| $\mathcal{B}$ | Mini-batch sampled from $\mathcal{D}$ during training |
| $Q(s,a)$ | Action-value function — estimated return from state $s$ taking action $a$ |
| $Q^*(s,a)$ | Optimal action-value function — true best possible return from $(s,a)$ |
| $Q^\pi_\varphi(s,a)$ | Q-function parameterized by $\varphi$, evaluated under policy $\pi$ |
| $\varphi$ | Parameters (weights) of the neural network representing $Q_\varphi$ |
| $J(\varphi)$ | Bellman error objective — the loss minimized during Q-function training |
| $\gamma$ | Discount factor $\in [0,1]$ — controls how much future rewards are valued |
| $T(\cdot \mid s_t, a_t)$ | Transition dynamics — probability of next state given current state and action |
| $d^{\pi_\beta}(s)$ | State visitation distribution — how often $\pi_\beta$ visits each state |
| $d^{\pi_{\text{off}}}(s)$ | State visitation distribution induced by the learned policy $\pi_{\text{off}}$ |
| $\mathbb{E}[\cdot]$ | Expected value operator |
| $\mathcal{H}(\cdot)$ | Entropy — measures spread/uncertainty of a distribution |
| $\mu(a \mid s)$ | Adversarial distribution over actions used in CQL penalty |
| $\alpha$ | Tradeoff factor — controls conservatism strength in CQL |
| $\mathcal{C}(\mathcal{B}, \varphi)$ | Conservative penalty term in CQL |
| $\mathcal{C}^0_{\text{CQL}}$ | Basic CQL penalty — provides pointwise lower bound on $Q^*$ |
| $\mathcal{C}^1_{\text{CQL}}$ | Refined CQL penalty — provides lower bound in expectation |
| $\mathcal{E}(\mathcal{B}, \varphi)$ | Standard Bellman error term in CQL objective |
| $\tilde{\mathcal{E}}(\mathcal{B}, \varphi)$ | Full CQL objective — Bellman error + conservative penalty |
| $R(\varphi)$ | Regularization term added to the Bellman error objective |
| $\propto$ | Proportional to |
| $\forall$ | For all — meaning the statement holds universally |
| $\notin$ | Does not belong to — e.g. $(s,a) \notin \mathcal{D}$ means unseen in dataset |
| $\sim$ | Sampled from — e.g. $a \sim \pi(\cdot \mid s)$ means $a$ is drawn from $\pi$ |
| $D_{\text{KL}}(\pi \| \pi_\beta)$ | KL divergence — measures how much $\pi$ deviates from $\pi_\beta$ |
