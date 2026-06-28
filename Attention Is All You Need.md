#architecture


*raw: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)*  
*reading date: 2026.4.20; Update: 2026.6.25  
Study Timeline pointer:  
last: [[word2vec]]; next: [[Improving Language Understanding by Generative Pre-Training]]*  

***Note that the following sections contain some techniques that are not directly covered in the raw paper.***  
*To be honest, transformer is quite complicated. I can't say I well comprehended its princples even now. So instead of giving reasoning and understandings, I will conduct this paper note by giving explicit data flow, step operations in it process, occasionally companied by some of my understanding to a certain section.*


# A. Data Flow & Step Operation
## 1. Tokenize
**input**: string/byte flow of the copra text  
**output**: V-dim one-hot vectors

subword-embedding(see: [[word2vec]])  
BPE, BBPE; re+BPE  
A pre-defined vocabulary size **V** will be obtained.  
After tokenization, each token can be represented as a V-dimension one-hot vector. Called $x_{init}$ .  
**Parameters**:  
	***model parameters***: none;  
	***hyper-paras***: {V}  

****
***Note that the following operations are conducted as matrixes, the GPU loads a batch of samples. For illustration, I will still use single vector to show the steps.***  
## 2. Embedding  
**input**: a V-dim one-hot vector
**output**: a $d_{model}$ embedding vector

The $x_{init}$ obtained in step 1 will be linearly transformed into a **$d_{model}$** embedding vector $x$:  
$$x=x_{init}^TW_e$$
where $W_e \in R^{V \times d_{model}}$.
**parameters**:  
	***model parameters***: {$W_e$};  
	***hyper-paras***: {V, $d_{model}$}

## 3. Transformer Layer
**input**: $d_{model}$ embedding vectors $X_i$, composed of n $x$, n could be the batch size. i means it comes from the $i_{th}$ Transformer Layer.
**output**: $d_{model}$ vectors $X_{i+1}$  

Each transformer layer consists of two sub-layers: Attention Sub-Layer and FFN Sub-Layer.

### 3.1 Attention Sub-Layer  

The input $x_0$ will be multiplied by 3 matrixes: $W_q, W_k, W_v$ , where $W_q, W_k \in R^{d_{model} \times k}$, $W_v \in R^{d_{model} \times v}$, to get 3 vectors Q, K and V:
$$Q=x_0^TW_q$$$$K=x_0^TW_k$$
$$V=x_0^TW_v$$
$k$ is a hyper-parameter which often takes the value equals $v$:  
$$v=\frac{d_{model}}{h}$$
and $h$ is another hyper-parameter denoting the number of "heads" in the multi-head architecture which will be explained later.  
All that being done, we can calculate the output of attention sub-layer:  
$$Attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt {d_k}})V$$
This is a $v$-dimension vector.  

#### *Unauthorized Understanding: what is QKV*
*Since all 3 matrixes are acquired in the training process, we can give then different meanings for understanding. Take a certain $Q_i$ as the **query message** of its input token represented by $x_i$ , standing for *what token $x_i$ would like to get*; $K_i$ as the **key message**, *standing for *what $x_i$ would like to **offer**, while $V_i$ stand for *what is  in $x_i$* .  
*The dot product $Q_iK_j^T$ measures the similarity between query message of token $i$ and offer message of token $j$ . That means, when $Q_iK_j^T$ is large, the target token $j$ has **what i wants**. So given a query token $i$, we caculate its $QK^T$ with all its context tokens and use softmax to generate a 1-in-total weight vector representing the contributions of $i_{th}$ token to the current token. Adding their $V \times weight$ together to get the representation vector of current token $j$.*  

### 3.2 Multi-head Attention(MHA)

There are $h$ parallel attention Sub-Layers in every transformer layer. So after the attention layer, we will get $h$ $v$-dimension vectors. concatenating them together we get a $d_{model}$-dimsension vector, given $v \times h=d_{model}$ .
To make the MHA layer more expressive, a $W_o \in R^{d_{model} \times d_{model}}$ will be applied to the concatenated vector: 
$$MHA_{out}=(Attention_1^T, Attention_2^T,...,Attention_h^T)W_o$$
 
#### *Unauthorized Understanding: MHA v.s. SHA*  
*MHA has the same parameter size of SHA(single head attention), but $h$ heads allows the co-existance of $h$ independent attention marks. After independent softmax, more information will be passed to the next layer. Take as an example:*  
$$x=(\textbf{10},5,3, \textbf{1},0.7,0.2)$$
*If we do an overall softmax to this vector, the dimension value $10, 5$ will stand out. But $1$ will almost vanish since it's too small compared to the previous ones. But if we adopt 2-head, then we get:*  
$$x_1=(10,5,4)$$ $$x_2=(1,0.7,0.2)$$
*And we can see that in $x_2$ the value 1 will also stand out, and due to mechanism of softmax, its activated value will be quite close to that of the 10 in $x_1$ . 
Through this way, more dimension values could be passed through to the next layer, making the overall MHA layer more expressive.*

### 3.3 FFN Sub-Layer

