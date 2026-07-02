
*raw: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)*  
*Read date: 2026.6.30 Update: 2026.6.30*  
*last: [[LLaMA-Open and Efficient Foundation Language Models]]*

# Review of activation functions

#### ReLU
(see: [[Attention Is All You Need]] #FFN);  
#### GELU(Gaussian Error Linear Unit):  
	*Used in BERT/GPT-2.*
The GELU has the form:  
$$GELU(x) = x \cdot \Phi(x)$$
where
$$\Phi(x) = P(X \le x) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$
where $X \sim N(0,1)$, so
$$\operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$$

#### SiLU & Swish
SiLU is defined as:
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
where $\sigma(x)$ is the standard Sigmoid function. SiLU is the $\beta=1$ example of Swish:
$$\text{Swish}_{\beta}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$

*Unauthorized Understanding: Comparison(ReLU, GELU, SiLU)*
	*GELU looks like ReLU in shape but has **non-zero gradent** in $(-\infty,0)$ compared with ReLU, and is smooth at $x=0$;  
	SiLU looks like GELU in shape but only contains an $e^x$ to calculate, which is much more easier that GELU.  
	A visualized comparison(AIGC):*  

![ReLU vs GELU vs SiLU comparison](NSM%28note%20support%20materials%29/img/Pasted%20image%2020260630142533.png)


# GLU Family
#QuickHop-SwiGLU

Standard GLU: 
$$\text{GLU}(x) = \sigma(xW + c) \otimes (xV + b)$$
To understand, take the "$\sigma(xW + c)$" as a probability $p$, ranging from $[0,1]$, while "$xV + b$" stands for the information $x$ carries. Then the GLU(x) can be intepretated as a process where the information that $x$ carries goes through a **filtering *Gate*** with a filter rate $p$.  

The form of the **Gate** does not always need to be Sigmoid. Combining with the differnet activation functions above, we have: 
- **ReGLU**: $\max(0, xW+b) \otimes (xV+c)$
- **GEGLU**: $GELU(xW+b)\otimes(xV+c)$
- **SwiGLU**: $Swish_{\beta}(xW+b)\otimes(xV+c)$


