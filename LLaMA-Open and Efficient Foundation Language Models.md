 #phenomenon #architecture
 >*𝘞𝘪𝘵𝘩𝘪𝘯 𝘢 𝘴𝘪𝘮𝘪𝘭𝘢𝘳 𝘵𝘳𝘢𝘪𝘯𝘪𝘯𝘨 𝘣𝘶𝘥𝘨𝘦𝘵, 𝘢 𝘸𝘦𝘭𝘭‑𝘵𝘳𝘢𝘪𝘯𝘦𝘥 𝘴𝘮𝘢𝘭𝘭 𝘮𝘰𝘥𝘦𝘭 𝘤𝘢𝘯 𝘰𝘶𝘵𝘱𝘦𝘳𝘧𝘰𝘳𝘮 𝘢𝘯 𝘶𝘯𝘥𝘦𝘳𝘵𝘳𝘢𝘪𝘯𝘦𝘥 𝘭𝘢𝘳𝘨𝘦 𝘰𝘯𝘦, 𝘸𝘪𝘵𝘩 𝘭𝘰𝘸𝘦𝘳 𝘪𝘯𝘧𝘦𝘳𝘦𝘯𝘤𝘦 𝘤𝘰𝘴𝘵𝘴 𝘢𝘵 𝘥𝘦𝘱𝘭𝘰𝘺𝘮𝘦𝘯𝘵.*

*raw: [[2302.13971v1]]*  
*reading date: 2026.6.26; update: 2026.6.26*  
*study timeline pointer:*  
last: [[Attention Is All You Need]], [[Improving Language Understanding by Generative Pre-Training]]  
next: [[GQA:Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints]]  


# Mindflow while Reading  
1. OpenAI Scaling Law(2020) failed because of the emergence of **inference**. *Small model infers faster*.([[Training Compute-Optimal Large Language Models]])  
2. Small model but larger training copra could still improve.  

# Modification to Transformer #lazytag

### 1. Pre-RMSNorm
*Paper ref: [[arXiv:1910.07467]]*
1. Calculate **Root Mean Square(RMS)**:
$$RMS(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
2. Normalize:
$$\hat{x}_i = \frac{x_i}{RMS(x)}$$
3. Scaling with out bias($\beta$) :
$$y_i = \hat{x}_i \cdot \gamma_i$$
Advantage v.s. traditional LN(*see [[Attention Is All You Need]])*:  
	**Less calculation cost**: Traverse the input $x$ only once;  
	**Less parameters**: Removing $\beta$ saves half the parameters in LN.  

*Undertanding: Why RMSNorm works?  
	Traditional LN used linear normalization to guarantee the **scaling invariance**, with which gradent could stay stable when layer outputs are on different scales. So scaling is more important than biasing. RMS kept this feature but did a mathamatical optimization.*  

### 2. SwiGLU
[[arXiv:2002.05202]]

### 3. RoPE
[[2104.09864v5]]

# Worth Noticing  

**Warmup**: Learning rate increase steadily towards target. To avoid gradient explosion at beginning when parameters are still random.  
**FlashAttention** #lazytag(need gpu architecture knowledge); Activation Checkpointing; Tensor Parallelism; Communication Overlap.  

**Traning Scaling:**![[Pasted image 20260626145557.png]]

#lazytag The **benchmarks**(might be used later in my research)