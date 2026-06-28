#architecture #pre-LN #decoder-only

*raw: [OpenAI — Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)*  
*reading date: 2026.6.25; Update: 2026.6.25*  
*study timeline pointer:   
last:[[Attention Is All You Need]]; next:[[Language Models are Few-Shot Learners]]*  

**Architecture: Decoder-Only (typical Transformer, *link* [[Attention Is All You Need]])**  

# Pre-train

Train on universal copra with goal Next Token Prediction to get parameters $\theta$: {$W_e(V*d_{model})$, Transformers($attention, ffn$)}


# Fine-Tuning

When given a given downstream task, for example: 2-classify:  
Add a task-specific head $W_j(d_{model} * j)$.  
During fine-tuning, GPT-1 keeps both LM head ($W_e^T$) and task head ($W_j$).  
When training, get both output with $W_e$ and $W_j$ as softmax of *next token predict* and softmax of *label predict*.  
Comparing both with its labels, we will get both $L_1$ and $L_2$，the goal is to minimize $L_3=L_2+ \lambda L_1$ , where $L_1$ and $L_2$ is calculated by cross-entropy.  
Note that during fine-tuning, both $W_j$ and $\theta$ will be optimized (full Fine-Tuning, different from later Linear Probe).  


# Something worth Noticing  

1. For small models whose $d_{model}$ is small, $W_e$ counts for large proportion of its parameters.  
2. GPT-1 used **aux LM loss** during fine-tuning (LM head + task head, joint loss $L_3=L_2+\lambda L_1$).  
3. **Pre-LN**: $y=x+F(LN(x))$, not $x+LN(F(x))$.  