## Euler's Formula( #AIGC)

### Definition
Euler's formula establishes the fundamental relationship between trigonometric functions and complex exponential functions:

$$e^{i\theta} = \cos\theta + i\sin\theta$$

For any real number $\theta$, $e^{i\theta}$ represents a point on the unit circle in the complex plane at an angle of $\theta$ radians from the positive real axis.

---

### Maclaurin Series Derivation
Recall the Taylor (Maclaurin) series expansions for $e^x$, $\cos x$, and $\sin x$ around $0$:

1. **Exponential function**:
   $$e^z = \sum_{n=0}^{\infty} \frac{z^n}{n!} = 1 + z + \frac{z^2}{2!} + \frac{z^3}{3!} + \frac{z^4}{4!} + \frac{z^5}{5!} + \dots$$

2. **Cosine function (even powers)**:
   $$\cos\theta = \sum_{n=0}^{\infty} \frac{(-1)^n \theta^{2n}}{(2n)!} = 1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \dots$$

3. **Sine function (odd powers)**:
   $$\sin\theta = \sum_{n=0}^{\infty} \frac{(-1)^n \theta^{2n+1}}{(2n+1)!} = \theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \dots$$

Substitute $z = i\theta$ into the exponential series, noting that $i^2 = -1, i^3 = -i, i^4 = 1$:

$$
\begin{aligned}
e^{i\theta} &= 1 + i\theta + \frac{(i\theta)^2}{2!} + \frac{(i\theta)^3}{3!} + \frac{(i\theta)^4}{4!} + \frac{(i\theta)^5}{5!} + \dots \\
&= 1 + i\theta - \frac{\theta^2}{2!} - i\frac{\theta^3}{3!} + \frac{\theta^4}{4!} + i\frac{\theta^5}{5!} - \dots
\end{aligned}
$$

Grouping the real and imaginary parts yields:

$$
\begin{aligned}
e^{i\theta} &= \left( 1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \dots \right) + i \left( \theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \dots \right) \\
&= \cos\theta + i\sin\theta
\end{aligned}
$$

---

### Key Properties & Forms Relevant to RoPE

To understand Rotary Position Embedding (RoPE), we leverage how Euler's formula links complex arithmetic to 2D geometric rotations.

#### 1. Complex Multiplication as 2D Rotation
Let a 2D vector $\mathbf{x} = [x_1, x_2]^T \in \mathbb{R}^2$ be represented as a complex number $z = x_1 + i x_2$. 

Multiplying $z$ by the complex rotation factor $e^{i\theta}$:

$$
\begin{aligned}
z e^{i\theta} &= (x_1 + i x_2)(\cos\theta + i\sin\theta) \\
&= (x_1\cos\theta - x_2\sin\theta) + i(x_1\sin\theta + x_2\cos\theta)
\end{aligned}
$$

Converting this result back to a 2D real vector yields the standard rotation matrix multiplication:

$$\mathbf{R}_{\theta} \mathbf{x} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

#### 2. Relative Position Encoding via Conjugation
RoPE injects relative positional information into the attention mechanism (dot product of Query and Key vectors). 

Let $q$ be a query at position $m$ rotated by $m\theta$, and $k$ be a key at position $n$ rotated by $n\theta$:
$$q_m = q e^{im\theta}, \quad k_n = k e^{in\theta}$$

In complex vector spaces, the inner product (dot product) corresponds to $\langle q_m, k_n \rangle = \text{Re}\{ q_m k_n^* \}$, where $*$ denotes the complex conjugate.

Applying Euler's identities:

$$
\begin{aligned}
q_m k_n^* &= (q e^{im\theta}) (k e^{in\theta})^* \\
&= q k^* e^{im\theta} e^{-in\theta} \\
&= q k^* e^{i(m-n)\theta}
\end{aligned}
$$

The resulting expression depends strictly on the **relative distance** $m - n$ rather than absolute positions. This mathematical convenience allows RoPE to naturally decay or oscillate as a function of distance in the attention matrix.