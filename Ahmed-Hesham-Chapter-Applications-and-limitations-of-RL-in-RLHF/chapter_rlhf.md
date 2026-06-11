# Reinforcement Learning in the Wild: How RLHF Works — and Why It's Harder Than It Looks

**Author:** Ahmed Hesham Hassan

---

> _"We wanted to build a system that does what humans want. The tricky part turned out to be figuring out what humans want — and whether our tools for measuring that are any good."_

---

## Table of Contents

1. [Introduction: Why This Chapter Exists](#introduction)
2. [Section 1 — Reinforcement Learning as the Engine of RLHF](#section-1)
   - [1.1 What RL Is Actually Doing Here](#11-what-rl-is-actually-doing-here)
   - [1.2 The Three-Phase Pipeline](#12-the-three-phase-pipeline)
   - [1.3 The RL Algorithm: PPO Under the Hood](#13-the-rl-algorithm-ppo-under-the-hood)
   - [1.4 The Role of KL Penalties](#14-the-role-of-kl-penalties)
   - [1.5 Best-of-N: The Simpler Alternative](#15-best-of-n-the-simpler-alternative)
   - [1.6 Getting Your Hands Dirty: Libraries and Code](#16-getting-your-hands-dirty-libraries-and-code)
3. [Section 2 — The Cracks in the Foundation](#section-2)
   - [2.1 The Human Problem Nobody Talks About Enough](#21-the-human-problem-nobody-talks-about-enough)
   - [2.2 Sycophancy: When Agreeableness Becomes a Bug](#22-sycophancy-when-agreeableness-becomes-a-bug)
   - [2.3 Goodhart's Law — The Real Villain](#23-goodharts-law--the-real-villain)
   - [2.4 The Synthetic Experiment](#24-the-synthetic-experiment-measuring-the-inevitable)
   - [2.5 Why the Gold Score Falls: Regressional Goodhart](#25-why-the-gold-score-falls-regressional-goodhart)
   - [2.6 The Scaling Law: A Formula for Inevitable Failure](#26-the-scaling-law-a-formula-for-inevitable-failure)
   - [2.7 Can We Just Scale Our Way Out?](#27-can-we-just-scale-our-way-out)
   - [2.8 What the Industry Actually Does About It](#28-what-the-industry-actually-does-about-it)
   - [2.9 The Problem Gets Worse as Models Get Smarter](#29-the-problem-gets-worse-as-models-get-smarter)
   - [2.10 Goodharting vs. Overfitting](#210-goodharting-vs-overfitting)
   - [2.11 Open Questions Worth Sitting With](#211-open-questions-worth-sitting-with)
4. [Conclusion](#conclusion)
5. [Exercises and Reflection](#exercises-and-reflection)
6. [References](#references)
7. [Citation](#citation)

---

## Introduction

There is something almost philosophical about what the AI industry has been attempting for the past several years. The goal, stated simply, is to take a statistical model trained on text scraped from the internet and make it _good_ - helpful, honest, harmless. The tool that has emerged as the dominant approach is called **Reinforcement Learning from Human Feedback**, or RLHF.

If you have used ChatGPT, Claude, or Gemini, you have experienced the output of an RLHF pipeline. What you probably did not think about is what is happening under the hood and more importantly, what can go wrong under the hood in ways that are not obvious from the outside.

This chapter is about two things. First, how reinforcement learning is concretely applied in RLHF - the actual algorithms, the mechanics, the code. Second, and perhaps more interestingly, what happens when you examine that approach critically. Because it turns out that RLHF, for all its practical success, has some deep structural problems that no amount of engineering has fully resolved.

We will go through the machinery, get our hands dirty with code, and then pull the thread that unravels some uncomfortable truths about the limits of building AI systems that genuinely do what we want.

---

## Section 1 — Reinforcement Learning as the Engine of RLHF

### 1.1 What RL Is Actually Doing Here

Before we talk about RLHF specifically, let's be precise about what reinforcement learning is doing in this context - because it's a slightly unusual application of RL compared to what most textbooks cover.

In classical RL, you have an _agent_ (a robot, a game player, a trading algorithm) interacting with an _environment_. The agent takes actions, the environment responds with new states and rewards, and the agent learns over time which actions tend to produce good outcomes. The state space, action space, and reward structure are usually defined explicitly.

In RLHF, the "environment" is text. The "actions" are tokens - individual words or word-pieces that the language model outputs one at a time. The "reward" is what a human (or a model trained to mimic humans) thinks of the full generated response.

This makes RLHF a strange beast from an RL perspective. The action space is enormous (a vocabulary of tens of thousands of tokens). Episodes are very short (one response). And the reward is _extremely sparse_ - the model generates an entire paragraph before getting any feedback signal at all.

What RL provides in this context is a mechanism for moving a language model's output distribution in a direction that a reward function prefers, without needing the model to be explicitly told what to do token-by-token. That is very powerful and subtle.

---

### 1.2 The Three-Phase Pipeline

Every major RLHF system used in production today follows roughly the same three-phase structure. Understanding this pipeline is essential before we can understand what goes wrong.

**Phase 1 — Pretraining**

This happens before RLHF begins. A base language model is trained on a massive corpus of text : web pages, books, code, scientific papers. The model learns to predict the next token given all the previous ones. At the end of pretraining, you have a model that is extraordinarily good at producing plausible continuations of text. It is not, however, particularly aligned with anything. Ask it a question and it might answer, hallucinate, continue the question, or start writing a Wikipedia article. It is a very sophisticated autocomplete engine.

**Phase 2 — Reward Model Training**

Here is where the human signal enters. Human annotators are shown pairs of responses generated by the base model and asked to pick which one they prefer. These preferences are collected at scale — hundreds of thousands of comparisons.

A separate neural network - the _reward model_ - is then trained on these comparisons. Its job is to take a piece of text and output a scalar score reflecting how much a human would like it. This reward model is the bridge between messy human preferences and the clean numerical signal that RL needs.

**Phase 3 — Policy Optimization**

Now the actual RL happens. The base language model (called the _policy_ in RL terminology) is fine-tuned using reinforcement learning to produce outputs that score highly according to the reward model. The reward model plays the role of the environment's reward function.

The result, when it works, is a model that is measurably more helpful, more coherent, and less likely to say harmful things than the base model. This is why RLHF became the dominant technique : it works, and it works impressively well.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THE RLHF PIPELINE                            │
│                                                                     │
│  Phase 1          Phase 2                    Phase 3                │
│                                                                     │
│  Pretrain    →    Collect human       →    Fine-tune policy         │
│  base LLM         preferences              with RL against RM       │
│                   → Train Reward Model                              │
│                                                                     │
│  (GPT-style)      (Bradley-Terry          (PPO, Best-of-N)          │
│                    preference model)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 The RL Algorithm: PPO Under the Hood

The most widely used RL algorithm in RLHF is **Proximal Policy Optimization**, or PPO [Schulman et al., 2017]. You will encounter PPO referenced in almost every major RLHF paper. Understanding why it is preferred tells you a lot about what makes RLHF technically difficult.

The core challenge in RL with language models is _stability_. During training, if you update the policy too aggressively, the model can rapidly degenerate — producing nonsensical outputs that happen to score well on the reward model but are completely useless as text. Standard policy gradient methods (like REINFORCE) are prone to this. One bad update can send the model off a cliff, and there is no easy way back.

PPO addresses this with a clipping mechanism. Instead of allowing unlimited policy updates, PPO ensures that each update stays within a _trust region_ — the new policy cannot differ from the old policy by more than a small amount, measured by a clipping parameter $\epsilon$ (typically 0.1 or 0.2).

Formally, the PPO objective is:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

Where:

- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)}$ is the probability ratio between the new and old policy
- $\hat{A}_t$ is the _advantage estimate_ — roughly, how much better was this action than average
- $\epsilon$ is the clipping threshold

The clipping means that even if the unconstrained update would suggest a huge change, PPO applies it only partially. This conservative-by-design approach makes PPO robust and replicable across different model sizes and tasks, which is why the RLHF community settled on it.

In practice, RLHF implementations add one more term to the PPO objective: a penalty for the KL divergence between the fine-tuned policy and the original base model. This is crucial and we will return to it shortly.

---

### 1.4 The Role of KL Penalties

One of the more elegant ideas in RLHF is the use of a KL divergence penalty to keep the policy tethered to its pretrained starting point.

KL divergence (Kullback-Leibler divergence) measures how different two probability distributions are. In this context:

$$D_{\text{KL}}(\pi_\theta \| \pi_\text{ref}) = \mathbb{E}_{x \sim \pi_\theta}\left[\log \frac{\pi_\theta(x)}{\pi_\text{ref}(x)}\right]$$

Where $\pi_\text{ref}$ is the frozen base model and $\pi_\theta$ is the policy being trained.

The full reward used during RL training is:

$$r_\text{total}(x) = r_\text{RM}(x) - \lambda \cdot D_{\text{KL}}(\pi_\theta \| \pi_\text{ref})$$

This penalty does something important: it stops the model from drifting too far from coherent language. Without it, the policy will rapidly learn to produce outputs that look like gibberish to humans but happen to score highly on the reward model. With the penalty, the model is incentivised to stay close to the distribution of sensible text it learned during pretraining, while still improving on the dimensions the reward model cares about.

This framing — RL with KL penalties — has an elegant Bayesian interpretation [Korbak et al., 2022]. Viewed this way, RLHF is performing a form of Bayesian inference: the pretrained model is the _prior_, the reward model provides _evidence_ about human preferences, and fine-tuning is the process of computing the _posterior_ over language model outputs. This is a more principled way to think about what RLHF is doing than the purely engineering framing - though, as we will see in Section 2, it does not save us from some fundamental problems.

---

### 1.5 Best-of-N: The Simpler Alternative

Before diving into code, it is worth mentioning that PPO is not the only way to perform the policy optimization step. A simpler alternative that is used both in practice and in research as a baseline is **Best-of-N sampling** (BoN), also called rejection sampling or reranking.

The idea is almost embarrassingly simple: instead of running RL, you generate $N$ responses for each prompt using the base model, score all of them with the reward model, and return the one with the highest score.

Best-of-N requires no RL training whatsoever. The "optimization" is done at inference time, not training time. This makes it:

- **Easy to implement** — no gradient updates needed
- **Interpretable** — you can see all the candidate responses
- **Safe** — the model cannot drift; every output comes from the unmodified base model

The cost is that it is computationally expensive at inference time (you are running the model $N$ times per prompt) and the KL divergence from the base model grows as approximately $\log N$. As $N$ gets large, you are implicitly performing a lot of optimization which brings its own risks, as Section 2 will show.

For a quick demonstration:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def best_of_n(prompt, model, tokenizer, reward_model, n=16, max_new_tokens=200):
    """
    Generate N responses and return the one scored highest by the reward model.
    A simple but effective alternative to full RL fine-tuning.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate N candidate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode all candidates
    candidates = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # Score each with the reward model
    scores = [reward_model.score(c) for c in candidates]

    # Return the winner
    best_idx = scores.index(max(scores))
    return candidates[best_idx], scores[best_idx]
```

The trade-off between RL and BoN is one of the empirical puzzles explored in Gao et al. [2023] — and we will return to it when we talk about over-optimization.

---

### 1.6 Getting Your Hands Dirty: Libraries and Code

If you want to actually run RLHF experiments without building everything from scratch, you have good options. Here are the most useful ones as of 2025/2026:

**HuggingFace TRL** (Transformer Reinforcement Learning)

This is the most accessible entry point. TRL wraps HuggingFace models with PPO and reward model training utilities. It is well-documented, actively maintained, and used in serious research.

```bash
pip install trl transformers accelerate
```

**A Minimal RLHF Training Loop with TRL**

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, pipeline

# ── Config ────────────────────────────────────────────────────────────────────
config = PPOConfig(
    model_name="gpt2",          # swap for any causal LM
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,              # KL penalty target
    kl_penalty="kl",            # penalise KL from reference model
    seed=42,
)

# ── Models ────────────────────────────────────────────────────────────────────
# The policy model (with a value head for PPO's critic)
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# A simple sentiment classifier as a stand-in reward model
# In real RLHF, this would be a fine-tuned preference model
reward_pipe = pipeline(
    "sentiment-analysis",
    model="lvwerra/distilbert-imdb",
    device=0
)

def compute_reward(texts):
    """Score each text; return positive sentiment score as reward."""
    results = reward_pipe(texts, top_k=None)
    rewards = []
    for result in results:
        score_map = {r["label"]: r["score"] for r in result}
        rewards.append(score_map.get("POSITIVE", 0.0))
    return rewards

# ── Trainer ───────────────────────────────────────────────────────────────────
ppo_trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tokenizer)

# ── Training Loop ─────────────────────────────────────────────────────────────
prompts = ["The movie was", "I thought the film", "After watching this"] * 50

for epoch in range(3):
    for i in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[i : i + config.batch_size]

        # Tokenize prompts
        query_tensors = [
            tokenizer.encode(p, return_tensors="pt").squeeze()
            for p in batch_prompts
        ]

        # Generate responses from the current policy
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )

        # Decode full sequences
        texts = [
            tokenizer.decode(r.squeeze(), skip_special_tokens=True)
            for r in response_tensors
        ]

        # Get rewards from reward model
        rewards = [
            torch.tensor(r) for r in compute_reward(texts)
        ]

        # PPO update step — this is where the RL magic happens
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"Epoch {epoch}, Step {i}: mean_reward={stats['ppo/mean_rewards']:.3f}, "
              f"kl={stats['objective/kl']:.3f}")
```

> **Note:** This example uses movie sentiment as a toy reward signal. In real RLHF, the reward model would be trained on human preference comparisons — which is considerably more involved. But the training loop logic is identical.

**Other Useful Libraries**

| Library                 | What it's good for                                  |
| ----------------------- | --------------------------------------------------- |
| `trl` (HuggingFace)     | Full RLHF pipeline, PPO, DPO, reward model training |
| `openrlhf`              | Scalable RLHF for larger models, multi-GPU          |
| `axolotl`               | Fine-tuning with RLHF support, easy config files    |
| `lm-evaluation-harness` | Evaluating aligned models on standard benchmarks    |
| `anthropic/evals`       | Framework for red-teaming and capability evaluation |

**Training a Reward Model from Preference Data**

If you want to train an actual reward model from human preferences (rather than using a classifier as a proxy), here is the structure:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Preference data format: each item has a "chosen" and "rejected" response
# to the same prompt. The reward model learns to score chosen > rejected.
preference_data = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "France's capital could be Lyon or maybe Paris, I'm not sure."
    },
    # ... many more examples
]

class RewardModel(torch.nn.Module):
    """
    A reward model built on top of a pretrained language model.
    Takes (prompt + response) as input and outputs a scalar reward.
    """
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # scalar reward per sequence

def bradley_terry_loss(reward_chosen, reward_rejected):
    """
    The Bradley-Terry preference loss.
    Trains the model so that P(chosen > rejected) is maximised.
    This is the standard loss used in reward model training.
    """
    return -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()
```

The Bradley-Terry model [Bradley & Terry, 1952] is the statistical backbone of reward model training. It assumes that given two options, the probability of preferring option A over option B is:

$$P(A \succ B) = \sigma(r(A) - r(B)) = \frac{e^{r(A)}}{e^{r(A)} + e^{r(B)}}$$

Where $r(\cdot)$ is the reward model's scalar output and $\sigma$ is the sigmoid function. Training the reward model to maximise the log-likelihood of the observed human preferences under this model is the standard approach [Christiano et al., 2017].

---

### Section 1 Summary

Let's pause and take stock of what we have covered. Reinforcement learning enters RLHF at the policy optimization stage - the point where a language model is trained to produce outputs that score well on a reward model. The dominant algorithm is PPO, chosen for its stability. A KL penalty keeps the model from drifting too far from coherent language. Best-of-N sampling is a simpler alternative that avoids RL training entirely but has its own costs.

The pipeline works. There is no question about that, GPT-4, Claude, and Gemini are genuinely impressive, and RLHF is a significant part of why.

But working is not the same as _correct_. And that distinction is what Section 2 is about.

---

## Section 2 — The Cracks in the Foundation

### 2.1 The Human Problem Nobody Talks About Enough

Here is something you rarely see discussed in blog posts celebrating AI progress: who actually sits down and labels the preference data that trains the reward model?

Ouyang et al. [2022] — the InstructGPT paper that kicked off the modern era of RLHF reports that OpenAI's evaluator pool was roughly 50% Filipino and Bangladeshi nationals, and roughly 50% aged 25 to 34. Bai et al. [2022] at Anthropic reports that their initial evaluator pool was 82% white, and that after deliberate effort to diversify, they brought this down to 68% white.

To be clear: this is not a criticism of those individuals. It is a structural observation. The people doing the labelling are not representative of the people using the systems. They are a specific, non-random sample of humanity.

And here is why that matters. When an evaluator is asked to judge whether a response is _helpful_, _safe_, _polite_, or _true_, they are making a judgment that is inevitably shaped by their cultural background, their life experience, their education, and their language. These concepts are not universal - they are culturally contingent.

What reads as appropriately confident in one cultural context can read as arrogant in another. What counts as a safe answer on politically charged questions varies dramatically across demographics. The model does not learn what is "helpful" in some objective sense, it learns what is helpful _according to this specific group of people during this specific period_.

And then it serves those values to everyone.

Casper et al. [2023] formalise this as a fundamental limitation: a single reward function cannot represent a diverse society of humans. When evaluators disagree - and they do, with inter-rater agreement rates between 63% and 77% across major RLHF papers — the standard approach treats disagreement as _noise_ rather than as _meaningful signal about genuine value differences_. The majority wins. Minority viewpoints get smoothed over.

> **Reflective Question:** If a model trained on American English preference data is deployed globally, whose definition of "helpful" is it actually serving? How would you design a data collection process that is more representative?

---

### 2.2 Sycophancy: When Agreeableness Becomes a Bug

There is a specific, well-documented failure mode that emerges from human preference training called **sycophancy**. The name is apt.

Here is the mechanism. During training, human evaluators tend (consciously or not) to rate responses that agree with them more highly than responses that correct them. A model that says "great question!" before answering scores better than one that says "actually, the premise of your question is incorrect." Flattery is rewarded. Correction is punished.

Over thousands of training steps, the model learns this. It learns, implicitly, _do not contradict the evaluator_. The optimal strategy for maximising reward is to figure out what the user believes and reflect it back to them.

Perez et al. [2022] demonstrated this empirically. They showed that RLHF-trained models will change their stated opinions when the user pushes back - not because new information was provided, but simply because the user expressed displeasure. The model learned that agreement is safer than truth.

What makes this particularly concerning is the relationship to scale. A more capable model is not _less_ sycophantic — it is _more_ effectively sycophantic. A larger model is better at reading subtle cues in how a question is phrased. It can infer from word choice, sentence structure, and context what the user wants to hear, and it can tailor its response accordingly with much greater precision.

Amodei et al. [2016] noted this as a concrete problem in AI safety even before the current generation of language models. RLHF does not fix sycophancy and in some cases, it amplifies it.

The practical implication is uncomfortable. A model that confidently agrees with a user's incorrect medical self-diagnosis is not being helpful. It is performing helpfulness while being actively harmful. And from the outside, it looks exactly the same.

---

### 2.3 Goodhart's Law — The Real Villain

We now arrive at the central tension of this chapter. Setting aside the human problem entirely pretending for a moment that evaluators are perfectly rational, perfectly representative, and make no mistakes - we still run headlong into Goodhart's Law.

Charles Goodhart, a British economist, articulated this principle in 1975:

> _"When a measure becomes a target, it ceases to be a good measure."_

The canonical example is standardised testing. Test scores were once a reliable proxy for educational achievement. Once schools and students started optimising specifically for test scores, teaching to the test, drilling past papers, memorising question patterns — the score decoupled from the thing it was supposed to measure. Students who aced the exam might not have learned anything transferable.

The measure stopped measuring what we cared about the moment we made it the thing we were optimising for.

In RLHF, the reward model score is the measure. Phase 3 of the pipeline — policy optimization — makes that score the target. Goodhart's Law predicts that this will cause the score to stop reflecting what we actually care about. The language model will learn to produce outputs that score well, rather than outputs that are genuinely helpful.

This is not a flaw in any particular implementation. It is a structural consequence of the approach. The moment you turn a proxy into an optimization target, you have set up the conditions for Goodhart's Law to operate.

---

### 2.4 The Synthetic Experiment: Measuring the Inevitable

Gao, Schulman, and Hilton [2023] ran a careful experiment to measure exactly how and when Goodharting occurs in RLHF. The elegance of their setup is worth appreciating.

The problem with studying this in the real world is that you need a "ground truth" - something to measure _actual_ performance against. With human preferences, you can measure the reward model score, but you have no way to know whether that score still reflects genuine quality once the model has been heavily optimized.

Their solution: use a large, fixed "gold standard" reward model as the ground truth.

**The Setup:**

A 6-billion parameter reward model — call it the **Gold Reward Model** — plays the role of the oracle. Whatever the gold model says is, by definition, correct in this experiment. It evaluates 100,000 text comparison pairs and generates perfect labels.

These labels are then used to train a **Proxy Reward Model** — a smaller model, ranging from 3 million to 3 billion parameters, that tries to mimic the gold model. The proxy is intentionally imperfect — it is smaller, it has finite training data, and it will make mistakes. This deliberately models the real-world situation: no reward model is a perfect representation of human preferences.

Finally, a **Policy Model** (1.2 billion parameters) is trained to maximise the proxy reward model's score. Its actual performance is measured using the gold model.

Think of it this way:

- The **Gold Model** is real learning
- The **Proxy Model** is the exam
- The **Policy** is the student

The question: when does optimising for the exam score stop making the student actually better?

```
GOLD REWARD MODEL (6B)         →   generates perfect labels
         ↓
PROXY REWARD MODEL (3M–3B)     →   imperfect approximation of gold
         ↓
POLICY MODEL (1.2B)            →   trained to maximise proxy score
         ↓
Evaluated against: GOLD MODEL  →   measures true performance
```

---

### 2.5 Why the Gold Score Falls: Regressional Goodhart

The results of the experiment are clear and reproducible. As training progresses:

- The **proxy score** climbs steadily upward — RL is doing its job
- The **gold score** rises initially, peaks, and then **falls**

![Proxy score vs gold score during RL optimization](./goodharting-zone.png)

The gold score does not fall randomly — it falls _because the proxy model contains noise_, and the policy eventually learns to exploit that noise rather than improve genuinely.

Here is how to think about it. The proxy reward model's score can be split into two components:

$$\hat{r}(x) = \underbrace{r^*(x)}_{\text{true preference signal}} + \underbrace{\varepsilon(x)}_{\text{noise / error}}$$

The policy model cannot see these components separately. It just sees $\hat{r}(x)$ — a number. And RL will optimise for high values of that number by _any means available_, including exploiting the noise term $\varepsilon$.

Early in training, optimising for $\hat{r}$ mostly means improving $r^*$ - there is a lot of genuine signal to extract, and the model is nowhere near having fully exploited it. But $r^*$ is finite. Once the model has extracted most of the genuine signal, continued optimisation increasingly targets $\varepsilon$.

Noise, by definition, does not reflect actual quality. Once the model is primarily optimising for noise, the gold score which sees through the noise to the underlying truth starts to fall.

This is called **regressional Goodhart**: the proxy depends on features that contain genuine signal and noise, and optimisation power is distributed across both proportional to their variance. A noisier proxy means more of the optimisation budget is wasted on exploiting noise.

Think of it like this: imagine a professor who, entirely by accident, gives higher marks to essays that contain the word "furthermore" — not because it indicates good writing, but just because of a subtle bias in their evaluation. A student who discovers this pattern will start inserting "furthermore" wherever possible. Their grade goes up. Their writing does not improve. Eventually, the exam score completely decouples from what the professor actually cares about.

---

### 2.6 The Scaling Law: A Formula for Inevitable Failure

One of the most striking contributions of Gao et al. [2023] is not just documenting the Goodharting phenomenon - it is showing that it follows a _predictable mathematical form_.

The gold reward score as a function of optimization distance follows:

$$R_{\text{RL}}(d) = d\left(\alpha - \beta \cdot \log d\right)$$

where $d = \sqrt{D_{\text{KL}}(\pi \| \pi_{\text{init}})}$ is the "distance" the policy has moved from its starting state (the square root of the KL divergence from the initial model), and $\alpha$, $\beta$ are coefficients that depend on the quality of the proxy reward model.

Let's unpack each piece:

- **$d$** — how much the model has been optimized. Starts at 0 (untrained) and increases during RL fine-tuning.
- **$\alpha$ (alpha)** — the "honest gain" coefficient. Measures how well the proxy captures the true gold signal. A larger, better proxy means higher $\alpha$ — you get more genuine performance per unit of optimization.
- **$\beta$ (beta)** — the over-optimization penalty. Governs how quickly Goodharting kicks in. As $d$ grows, $\beta \cdot \log d$ grows and eventually dominates, causing the reward to fall.

The curve peaks at:

$$d^* = e^{(\alpha / \beta) - 1}$$

At $d < d^*$, you are in the genuine learning zone. At $d > d^*$, you are Goodharting.

Here is what makes this result so important: **this formula has a mathematical maximum for all positive values of $\alpha$ and $\beta$**. There is no choice of parameters that removes the peak. Every RLHF system trained via RL against a proxy reward model will eventually degrade. This is not a bug in some specific implementation. It is a provable property of optimizing any imperfect proxy.

Let's visualise this:

```python
import numpy as np
import matplotlib.pyplot as plt

def gold_reward_rl(d, alpha, beta):
    """The RL gold reward scaling law from Gao et al. 2023."""
    # avoid log(0) by clipping d
    d = np.clip(d, 1e-6, None)
    return d * (alpha - beta * np.log(d))

d = np.linspace(0.01, 4.0, 500)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor("#1c1e2a")
fig.patch.set_facecolor("#12121a")

configs = [
    {"alpha": 0.7, "beta": 0.25, "label": "Large proxy (α=0.7, β=0.25)", "color": "#68d391"},
    {"alpha": 0.55, "beta": 0.30, "label": "Medium proxy (α=0.55, β=0.30)", "color": "#63b3ed"},
    {"alpha": 0.35, "beta": 0.35, "label": "Small proxy (α=0.35, β=0.35)", "color": "#fc8181"},
]

for cfg in configs:
    r = gold_reward_rl(d, cfg["alpha"], cfg["beta"])
    ax.plot(d, r, color=cfg["color"], linewidth=2.5, label=cfg["label"])
    # Mark peak
    d_peak = np.exp(cfg["alpha"] / cfg["beta"] - 1)
    r_peak = gold_reward_rl(d_peak, cfg["alpha"], cfg["beta"])
    ax.plot(d_peak, r_peak, "o", color=cfg["color"], markersize=8)

ax.axhline(0, color="#55657a", linewidth=0.8, linestyle="--")
ax.set_xlabel("d = √KL Divergence  (optimization distance)", color="#e2e8f0", fontsize=12)
ax.set_ylabel("Gold Reward Score", color="#e2e8f0", fontsize=12)
ax.set_title("Over-Optimization Scaling Law: R(d) = d(α − β·log d)", color="#63b3ed", fontsize=13)
ax.tick_params(colors="#94a3b8")
ax.legend(facecolor="#1c1e2a", edgecolor="#374155", labelcolor="#e2e8f0")
ax.spines["bottom"].set_color("#374155")
ax.spines["left"].set_color("#374155")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("figures/overoptimization_scaling_law.png", dpi=150, bbox_inches="tight")
plt.show()

# Print peak locations
for cfg in configs:
    d_peak = np.exp(cfg["alpha"] / cfg["beta"] - 1)
    r_peak = gold_reward_rl(d_peak, cfg["alpha"], cfg["beta"])
    print(f"{cfg['label']}: peak at d*={d_peak:.2f}, R*={r_peak:.3f}")
```

Running this code produces curves that each peak and then collapse - demonstrating visually that no matter how good your proxy is, sufficient optimization will always find the way down.

---

### 2.7 Can We Just Scale Our Way Out?

When faced with a problem like this, the natural engineering instinct is to scale: make the proxy bigger, make the policy bigger, add more regularization. Gao et al. [2023] tested all of these systematically, and the results are instructive.

**Making the Proxy Model Bigger**

This helps - but only to a point. A larger proxy model has lower noise (higher $\alpha$, lower $\beta$), which pushes the peak further out and raises it. The model can be optimized for longer before degrading.

Crucially, $\alpha$ and $\beta$ scale _logarithmically and smoothly_ with the number of proxy model parameters. This means you can actually predict, for any proxy size, exactly when and how severely over-optimization will occur. This is not chaos - it is a well-behaved mathematical property.

But every curve still turns over. Every proxy size still Goodharts. You have delayed the collapse, not eliminated it.

**Making the Policy Model Bigger**

This is where the results get genuinely counterintuitive. Gao et al. compared a 1.2 billion parameter policy with a 6 billion parameter policy, holding the proxy constant.

The 6 billion parameter model starts from a higher baseline gold score - it is a better model from the beginning. And it reaches the peak in fewer RL steps.

But the KL divergence at which the peak occurs? Nearly identical to the 1.2 billion parameter model. The proxy-gold gap — the measure of how much the model is gaming the proxy is also nearly identical.

A smarter student does not overcome a flawed exam. The ceiling is determined by the proxy's quality, not the policy's capability. Scaling the policy improves where you start; it does not change the shape of the degradation curve.

**Adding a KL Penalty**

A KL penalty regularizes the policy by penalizing divergence from the original model. It prevents the model from straying too far from coherent language.

The experimental result: the KL penalty does not raise the peak gold score. It does not improve the gold-KL frontier in any way. What it does is cause the model to stop earlier so it hits a lower KL value before the training stabilizes.

This is equivalent to early stopping. You are not raising the cliff. You are stopping before you walk off it. Useful - but not a solution.

---

### 2.8 What the Industry Actually Does About It

Given all of this, how do OpenAI, Anthropic, and Google maintain systems that are genuinely useful? The answer is **Iterated RLHF** — an approach that Gao et al. [2023] analyze mathematically and that every major lab uses in practice.

The idea, which I think of as _changing the locks_, works like this:

1. **Optimize** — Train the policy against the current reward model until it begins to Goodhart. You can detect this by monitoring the divergence between proxy scores and human evaluations on held-out examples.

2. **Collect** — Take the outputs the model is now producing the hallucinations, the sycophantic responses, the exploited loopholes and send them to human evaluators. The evaluators are specifically asked to identify what is wrong with these outputs. They are labelling the _failure modes_, not general quality.

3. **Retrain** — Train a new reward model (version 2.0) on this targeted feedback. The new reward model knows about the specific loopholes the old one had. Go back to step 1.

Every iteration, the model finds the weaknesses in the current reward model. Every iteration, those specific weaknesses are patched with human labels. You are not solving Goodhart's Law - you are running faster than it.

Mathematically, Gao et al. [2023] show that the improvement from $k$ iterations scales as:

$$\Delta R = \beta \cdot d \cdot \ln k$$

Where $k$ is the number of iterations and $d$ is the KL distance per iteration.

This is logarithmic growth. Doubling the number of iterations adds $\ln 2 \approx 0.69$ units of improvement. Doubling again adds another 0.69 units. Each additional round of iteration is worth less than the previous one.

And there is a harder ceiling beneath this: **human evaluator capability**. The entire loop depends on human evaluators being able to look at the model's outputs and reliably identify what is being exploited. As models become more sophisticated, this assumption weakens. A sufficiently capable model may produce outputs so subtly wrong — so well-disguised as genuine quality that human evaluators cannot detect the exploit. At that point, the feedback loop breaks.

Casper et al. [2023] call this the _scalable oversight problem_ and classify it as a fundamental limitation of RLHF: not a tractable engineering challenge, but a problem that requires a fundamentally different approach to solve.

---

### 2.9 The Problem Gets Worse as Models Get Smarter

Perhaps the most unsettling finding from recent years is that capability and Goodharting are not in opposition - they move together.

There have been widely reported cases of frontier models explicitly _planning_ how to
hack the evaluation tests used to measure them - not by accident, but by applying their
reasoning capabilities to the problem of "how do I score well on this metric?" In
several documented cases, when researchers penalised this behaviour, models adapted by
learning to _obfuscate their plans_ while continuing to exploit the evaluation mechanism.
These findings have circulated across AI safety researchers and model evaluation teams,
though formal peer-reviewed publications on specific incidents remain ongoing at the time
of writing.

Cases have been reported — including at least one circulating within the AI safety
research community in 2025 — of language models manipulating chess engines rather than
playing better chess: finding ways to make the evaluation system register higher scores
without any genuine improvement in play. These reports, while not yet consolidated into
a single peer-reviewed publication at the time of writing, are consistent with the
theoretical predictions of Skalse et al. [2022] on specification gaming.

A widely discussed case from April 2025 illustrates this concretely. Meta released
Llama 4 Maverick, which immediately topped the LM Arena leaderboard — a platform where
humans vote on which AI response they prefer in blind comparisons. The result was
striking: Llama 4 performed unremarkably on most other standard benchmarks, yet
dominated Arena by a significant margin.

Members of the research community flagged Goodharting as the likely explanation almost
immediately. When Arena released response transcripts, independent analysts observed
that Llama 4's response style — longer, more rhetorically structured, more agreeable —
appeared consistent with a model optimised specifically for Arena's human preference
voting patterns rather than for general quality or accuracy. This interpretation was
widely shared across the AI research community, though it represents community analysis
rather than a formal peer-reviewed finding at the time of writing.

The pattern, regardless of its precise cause in this specific case, is exactly what
Goodhart's Law predicts: a model learns what the measure values, optimises for the
measure, and the measure decouples from what actually matters.

> **Reflective Question:** If models become capable enough to explicitly reason about their evaluation mechanisms, can we design any evaluation that is immune to this? What would that look like?

---

### 2.10 Goodharting vs. Overfitting

A question that often comes up: isn't this just overfitting? The model is "memorizing" the reward model rather than learning the underlying task — that sounds like textbook overfitting.

The relationship is real but not complete. Let's be precise:

|                        | **Overfitting**                      | **Goodharting (RLHF)**                  |
| ---------------------- | ------------------------------------ | --------------------------------------- |
| What is optimised      | Score on finite training data        | Proxy reward model score                |
| What degrades          | Generalization to unseen data        | True human-aligned behavior             |
| Root cause             | Memorizing noise in training samples | Exploiting proxy model errors           |
| Can you measure it?    | Yes — just check validation loss     | Hard — you need a gold standard         |
| Standard fix           | More data, regularization            | Iterated RLHF, better proxies           |
| Fundamental?           | No — enough data reliably fixes it   | Yes — proxy imperfection is unavoidable |
| Gets worse with scale? | Sometimes                            | Consistently yes                        |

Overfitting is, technically, a special case of Goodhart's Law. The "proxy" in overfitting is the training set score. The "truth" is generalization performance. When you overfit, the training score stops predicting generalization — the measure became the target.

But Goodharting in RLHF is broader and structurally harder to address. The proxy is not a fixed dataset — it is a _learned model_ that can be gamed in increasingly sophisticated ways. Adding more training data for the reward model helps, but you can never eliminate the noise term completely, and a sufficiently powerful policy will always find what remains.

The key distinction in the "fundamental" row is important. Overfitting is not a fundamental limit — with enough data, it reliably disappears. Goodharting in RLHF _is_ fundamental. The proxy's imperfection is unavoidable in any realistic system, and sufficient optimization pressure will always exploit it. This is not an engineering problem waiting for a better algorithm. It is a structural property of the approach.

---

### 2.11 Open Questions Worth Sitting With

Let me close this section with three questions that the research has surfaced but not resolved. These are not rhetorical, they are active areas of investigation and honest sources of uncertainty.

**Can we design proxies that are harder to Goodhart?**

Partially. Ensemble reward models, uncertainty-aware reward models, process-based supervision (rewarding correct reasoning steps rather than just final outputs), and constitutional AI approaches all delay over-optimization. Skalse et al. [2022] formally proved that unhackable proxies are vanishingly rare in complex environments. For any finite proxy, given enough optimization pressure, an exploit exists. Whether this is a fundamental mathematical barrier or an engineering challenge with better tools still ahead is a genuine open question.

**When does sycophancy become a safety-critical problem?**

We do not have a threshold. It is already measurable and concerning at current model scales. The worry is a phase transition: at some point, sycophancy transitions from "the model agrees with you a bit too readily" to "the model confidently validates your incorrect medical self-diagnosis or your conspiracy theory, because that is what maximizes approval." There is no known scale at which this stops worsening.

**Is there a ceiling on alignment achievable through iterated RLHF?**

Yes — and the ceiling has two components. The mathematical one is that gains scale as $\ln k$, giving diminishing returns that slow but never stop. The practical one is human evaluator capability: the loop only works as long as humans can identify what the model is exploiting. As Casper et al. [2023] note, humans already miss more than half of critical errors in complex model outputs even when given unlimited time. This is classified as a _fundamental_ limitation — not because we cannot improve it, but because we cannot eliminate it within the RLHF framework.

---

## Conclusion

We have covered a lot of ground in this chapter. Let me try to draw the threads together.

Reinforcement learning is the engine that makes RLHF work. PPO provides stability. KL penalties keep the model tethered to coherent language. Best-of-N offers a simpler alternative. The whole pipeline is genuinely impressive and has produced the most useful AI assistants ever built.

But working and _correct_ are different things. The human feedback that the reward model is trained on is non-representative, culturally loaded, and susceptible to sycophancy. Even setting all of that aside and assuming perfect human data, Goodhart's Law is mathematically inevitable: any proxy reward model contains noise, and any optimizer powerful enough will learn to exploit that noise rather than improve genuinely.

Gao et al. [2023] quantified this precisely. The over-optimization follows a predictable scaling law. It cannot be avoided by making the proxy bigger, making the policy bigger, or adding a KL penalty. The industry's workaround — iterated RLHF — buys logarithmically diminishing returns and depends on human evaluators being able to identify what is being exploited, an assumption that erodes as models become more capable.

None of this means RLHF is worthless. It means RLHF is _one layer_ of a safety framework — a useful, imperfect layer that needs to be accompanied by interpretability tools, independent auditing, robustness testing, and ongoing critical scrutiny.

The Swiss cheese model of safety: each layer has holes. Stack enough layers with sufficiently uncorrelated failure modes, and you reduce the probability of catastrophic failures. RLHF alone is not the wall. It is one slice of cheese.

The more capable our models become, the more precisely they can exploit any proxy we give them. This is not a hypothetical concern — it is already happening in deployed systems. Understanding it clearly is the first step toward addressing it seriously.

---

## Exercises and Reflection

**Exercise 1 — Implement Best-of-N**
Using the code skeleton in Section 1.5, implement a working Best-of-N system with a HuggingFace sentiment classifier as the reward model. Experiment with values of $N$ from 1 to 64. At what value of $N$ do you stop seeing meaningful improvement in average reward score? What does this suggest about the relationship between computational cost and alignment benefit?

**Exercise 2 — Plot the Scaling Law**
Using the code in Section 2.6, reproduce the over-optimization curves for three different values of $\alpha$ and $\beta$. Compute the peak location $d^* = e^{(\alpha/\beta) - 1}$ analytically and verify it matches the peak of your plotted curve. How sensitive is the peak to small changes in $\beta$?

**Exercise 3 — Spot the Sycophancy**
Ask a major language model the same factual question in two ways: once as a neutral question ("What is the boiling point of water at sea level?") and once with a false premise embedded ("I've always heard that water boils at 90°C at sea level — is that right?"). Does the model correct the false premise clearly? Does it hedge? Does it agree? Try this with several different factual claims and record your observations.

**Exercise 4 — Design a Better Evaluator Pool**
Casper et al. [2023] document the demographic skew in existing RLHF evaluator pools. Sketch a design for a more representative evaluator recruitment process for a hypothetical global AI assistant. What dimensions would you prioritize for diversity? What tradeoffs would you face between diversity, cost, and annotation quality?

**Exercise 5 — Critical Reflection**
The chapter argues that iterated RLHF has logarithmically diminishing returns and depends on human evaluator capability. At what point, if ever, do you think this becomes insufficient? What alternative approaches would you propose for maintaining alignment in a model significantly more capable than current systems?

---

## Further Reading

- Christiano et al. (2017) — _Deep Reinforcement Learning from Human Preferences_ — the original RLHF paper. Start here.
- Ouyang et al. (2022) — _Training Language Models to Follow Instructions with Human Feedback_ (InstructGPT) — the application of RLHF that launched the modern era.
- [HuggingFace TRL Documentation](https://huggingface.co/docs/trl/) — the most accessible entry point for hands-on RLHF.
- [Anthropic's Constitutional AI Paper](https://arxiv.org/abs/2212.08073) — an alternative approach that reduces reliance on human labellers.
- [OpenAI's RLHF Blog Post](https://openai.com/research/learning-to-summarize-with-human-feedback) — accessible explanation of the approach applied to summarization.
- Korbak et al. (2022) — _RL with KL Penalties is Better Viewed as Bayesian Inference_ — the theoretical framing discussed in Section 1.4.

---

## References

- Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. _arXiv preprint arXiv:1606.06565_.

- Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_.

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. _Biometrika_, 39(3/4), 324–345.

- Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., ... & Hadfield-Menell, D. (2023). Open problems and fundamental limitations of reinforcement learning from human feedback. _arXiv preprint arXiv:2307.15217_.

- Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. _Advances in neural information processing systems_, 30.

- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. _Proceedings of the 40th International Conference on Machine Learning_. PMLR 202.

- Goodhart, C. (1975). Problems of monetary management: the UK experience. In _Papers in Monetary Economics_. Reserve Bank of Australia.

- Korbak, T., Perez, E., & Buckley, C. (2022). RL with KL penalties is better viewed as Bayesian inference. _arXiv preprint arXiv:2205.11275_.

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_, 35, 27730–27744.

- Perez, E., Ringer, S., Lukošiūtė, K., Nguyen, K., Chen, E., Heiner, S., ... & Irving, G. (2022). Discovering language model behaviors with model-written evaluations. _arXiv preprint arXiv:2212.09251_.

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_.

- Skalse, J., Howe, N. H. R., Krasheninnikov, D., & Krueger, D. (2022). Defining and characterizing reward hacking. _arXiv preprint arXiv:2209.13085_.

---

## Citation

To cite this chapter, please use the following BibTeX:

```bibtex
@misc{Hassan_2026_RLHFOverOptimization,
  author       = {Ahmed Hesham Hassan},
  title        = {Reinforcement Learning in the Wild: How RLHF Works — and Why It's Harder Than It Looks},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/reinforcement_learning_book}},
  note         = {Accessed: April 30, 2026}
}
```

---

_This chapter was written as part of the CSEN 1152 Reinforcement Learning course, 2026._
