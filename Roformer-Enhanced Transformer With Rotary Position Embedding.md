
*raw: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)*  
*Reading date: 2026.6.28; Update: 2026.6.29*  
*last: [[LLaMA-Open and Efficient Foundation Language Models]]*


In traditional transformer, positional information of tokens are represented by adding a pre-generated absolute position encoding directly to the entrance of the **first** transformer layer. This is called Absolute Positional Encoding(**APE**, detail see [[Attention Is All You Need]], #APE).  

Shaw **RPE** [arXiv:1803.02155](https://arxiv.org/abs/1803.02155), introduced adding a relative PE to K and V(see eq5 in raw paper).  
Dai et al.[arXiv:1901.02860](https://arxiv.org/abs/1901.02860) decomposed the *Query* operation in traditional transformer with APE, which can be viewed as four parts: *$word_i$ meaning 2 $word_j$ meaning*; *$word_i$ meaning to $word_j$ position*; *$word_i$ position 2 $word_j$* meaning and *$pos_i$ 2 $pos_j$*. With this understanding, several different forms of RPE has been proposed(see raw eq6-10), and eq10 has proved to be the most efficient:  
$$\boldsymbol{q}_m^{\top} \boldsymbol{k}_n = \boldsymbol{x}_m^{\top} \boldsymbol{W}_q^{\top} \boldsymbol{W}_k \boldsymbol{x}_n + \boldsymbol{x}_m^{\top} \boldsymbol{W}_q^{\top} \boldsymbol{W}_k \tilde{\boldsymbol{p}}_{m-n} + \tilde{\boldsymbol{p}}_{m-n}^{\top} \boldsymbol{W}_q^{\top} \boldsymbol{W}_k \boldsymbol{x}_n$$
where $p_{m-n}$ is from a trainable PE table, usually with the relative distance between a fixed $[-r, r]$ range, r being a hyper-parameter.  

*Unauthorized Understanding: Shaw RPE is a feature improving engineering, replacing the absolute position i and j with a one-dimension but more relevant **relative distance**; And the later work on RPE is to decide which of the 4 items in eq6 is irrelavant and can be removed to enhance the PE performance.*  

## RoPE
#QuickHop-RoPE 
*Quick Patch: [[Math prerequisites]] #EulersFormula*  

*Since RoPE is quite complicated. Let's begin by first giving a through def and necessary intermediate mathematic proving.*  

Suppose that $d_h$ is even. $W_qx_m$ generates a $d_h$ dimension vector. The rotation matrix $R_{1,m}$:
$$
R_{1,m}=\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 \\
\sin m\theta_1 & \cos m\theta_1
\end{pmatrix}
$$
can rotate a 2 dimension vector by $m\theta_1$ degree. 
Consider this block diagonal matrix:
$$R_{\Theta,m}^{d_h}=\begin{pmatrix}
R_{1,m} &  &  & \\
 & R_{2,m} & & \\
 & & ... & \\
 & & & R_{\frac{d}{2},m}
\end{pmatrix}$$
Where $\theta_i$ shares the definition with APE(see: [[Attention Is All You Need]] #APE).  
$R_{\Theta,m}^{d_h}$ satisfies:
$$(R_{\Theta,m}^{d_h})^TR_{\Theta,n}^{d_h}=R_{\Theta,n-m}^{d_h}$$
To prove, consider $R_m,R_n=R_{1,,m},R_{1,n}$ ,with direct matrix multiplication, we can verify that $R_{1,m}^TR_{1,n}=R_{1,n-m}$ . Then using the feature of block matrix, the target feature can be proved.  
With this, look at the QK process in transformer with RoPE where: 
$$q_m=R_{\Theta,m}^{d_h}W_qx_m$$
$$k_n=R_{\Theta,n}^{d_h}W_kx_n$$
That's a multiplicative PE, different from APE which is additive.  
And with the previously proved feature, we can prove: 
$$
\boldsymbol{q}_m^{\mathsf{T}} \boldsymbol{k}_n = (\boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_q \boldsymbol{x}_m)^{\mathsf{T}} (\boldsymbol{R}_{\Theta,n}^d \boldsymbol{W}_k \boldsymbol{x}_n) = \boldsymbol{x}_m^{\mathsf{T}} \boldsymbol{W}_q^{\mathsf{T}} \boldsymbol{R}_{\Theta,n-m}^d \boldsymbol{W}_k \boldsymbol{x}_n
$$
which shows that RoPE rightly carries the information of relative position.

Note that RoPE is usually applied to every transformer layer, different from APE which is applied only at the first transformer layer, since RoPE only participates QK process while APE also acts on V. 