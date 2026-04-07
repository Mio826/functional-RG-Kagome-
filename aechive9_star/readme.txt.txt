this version has a major issue:
the patch representation is wrong.
好，这一步现在可以**比较严格地钉死**了。

结论先说：

[
\sum_k \quad \Longrightarrow \quad \sum_{\text{patch }p}; 1\times(\text{patch representative bubble})
]

也就是你现在这套 flow 里，**那个问号既不是 patch 长度，也不是 patch 长度除以 (v_F)，更不是某种真正的 patch 内积分平均**。
它本质上就是：

* 每个 patch 取一个代表点；
* 通过 partner map 找到 (p+Q) 或 (Q-p) 的 partner patch；
* 用这两个代表点的能量 (\varepsilon_p,\varepsilon_{p'}) 去算 bubble；
* 然后把这个 bubble 当作该 patch 的 loop 权重。

所以更准确地写，是：

[
\sum_k F(k,Q)
;;\not\approx;;
\sum_p \ell_p,F(k_p,Q)
]

也不是

[
\sum_p \frac{\ell_p}{|v_F(k_p)|}F(k_p,Q),
]

而是更接近

[
\boxed{
\sum_p F_{\text{bubble}}!\bigl(\varepsilon_p,\varepsilon_{\text{partner}(p,Q)}\bigr)
}
]

其中每个 patch **默认等权**，差别只来自 bubble 的能量值本身。这个结论可以直接从你代码里读出来。 

---

# 一、先从连续理论写起：真正的 one-loop 积分应该长什么样

在连续动量的温度流 fRG 里，右手边 one-loop 的本质是

[
\Phi \sim \int_{\text{BZ}} \frac{d^2k}{(2\pi)^2}; L(k,Q); \Gamma\Gamma,
]

或者在贴近费面的近似里，把二维积分写成“沿费面切向 + 垂直费面法向”的坐标：

[
d^2k = dk_\parallel,dk_\perp.
]

如果再把 (k_\perp) 的那部分先做掉，通常会留下一个沿费面的积分：

[
\Phi \sim \int_{\text{FS}} d\ell; \frac{1}{|v_F(\ell)|}; \widetilde L(\ell,Q); \Gamma\Gamma.
]

所以在**严格物理意义**上，patch 离散以后最自然的近似应该是

[
\int_{\text{FS}} d\ell; \frac{1}{|v_F|}(\cdots)
;;\longrightarrow;;
\sum_p \frac{\ell_p}{|v_F(p)|}(\cdots),
]

其中 (\ell_p) 是 patch 覆盖的费面弧长。

这就是你刚才问的那几个候选项为什么是合理候选：

* 1
* patch 长度
* patch 长度除以 (v_F)

因为从连续理论看，真正自然的是**后两者那一类**，不是单纯的 1。

---

# 二、再看 RMP 对 N-patch 的描述：它确实是“patch 常值近似”，但 loop 里仍然是在近似连续积分

RMP 对 N-patch 的表述是：

* 把耦合函数在 patch 上近似成常数；
* 外腿 (k_1,k_2,k_3) 取 patch 中心、也就是费面上的代表点；
* loop 里需要把 (k_4) 也投影回 patch。 

注意这段话的关键是：

> 它是在**近似连续积分**，不是把问题改成“纯 patch 标签等权求和模型”。

也就是说，N-patch 的精神是“patchwise constant approximation to a momentum integral”，不是“每个 patch 一个样本点，样本点等权”。

所以如果你代码最后真的落成“每个 patch 一个代表点 + 等权求和”，那它和标准 N-patch 的语义就已经有距离了。

---

# 三、现在直接看你代码：真正的“问号”是什么

## 1. flow 的 internal cache 是怎么构造的

`frg_kernel.py` 里 `build_pp_internal_cache_vec` 和 `build_ph_internal_cache_vec` 的核心逻辑非常清楚：

* 先取 source patch 集的能量数组 `eps_a = _patch_energies(psa)`
* 再取 target patch 集的能量数组 `eps_b_all = _patch_energies(psb)`
* partner map 给出每个 patch (a) 的 partner patch (b)
* 然后权重直接设成

pp:
[
w_a = \texttt{_bubble_dot_pp_vec}(\varepsilon_a,\varepsilon_b)
]

ph:
[
w_a = \texttt{_bubble_dot_ph_vec}(\varepsilon_a,\varepsilon_b)
]

代码里没有任何 patch 长度、(v_F)、Jacobian 因子乘进去。 

这就已经把问题钉死了一大半：

[
\boxed{
\text{flow 的 patch 权重不是 } \ell_p,\ \ell_p/|v_F|,\ \text{而只是 bubble}(\varepsilon_p,\varepsilon_{p'})。
}
]

---

## 2. `_patch_energies` 真的只是 patch 代表点能量

`instability.py` 里 `_patch_energies(ps)` 就是

[
\varepsilon_p = \text{patch object 里存的 } p.\text{energy}
]

没有任何 patch 宽度平均，也不是 patch 内积分。

所以这里的 bubble 不是

[
\int_{\text{patch }p} dk_\parallel,dk_\perp; G G
]

而只是

[
GG \text{ 在两个代表点能量上的 Matsubara 和。}
]

---

## 3. 这些 internal cache 真的被原封不动送进 RHS

`frg_flow.py` 里 `_refresh_cache_weights` 会对每个 transfer (Q) 构造：

* `pp_internal_by_iq`
* `ph_internal_by_iq`
* `phc_internal_by_iq`

其中每个 cache 只有三样东西：

* `partner`
* `residual`
* `weights`

然后 `compute_vertex_rhs` 里直接把这些 cache 喂给

* `compute_pp_vertex_contribution_sz0`
* `compute_phd_vertex_contribution_sz0`
* `compute_phc_vertex_contribution_sz0`。

而在 `frg_kernel.py` 里，这三个 one-loop 贡献都是形如：

[
\sum_a w_a \times (\text{两个或三个 }V\text{ 的乘积})
]

这里的 (a) 就是 patch index。也就是说，离散 loop 的数学形式就是

[
\boxed{
\Phi(Q) \sim \sum_{a\in \text{patches}} w_a(Q); \mathcal T_a
}
]

其中

[
w_a(Q)=\text{bubble}\bigl(\varepsilon_a,\varepsilon_{b(a,Q)}\bigr).
]

没有额外 patch measure。

---

# 四、所以“问号”到底是什么？现在可以明确写出来

你刚才问的是：

[
\sum_k \to \sum_{\text{patch }p}; ?
]

现在答案是：

[
\boxed{
? ;=; 1
}
]

但这句话要说完整，不然会有歧义。

更严格地说，不是“单纯求和 1”，而是：

[
\boxed{
\sum_k(\cdots)
\ \leadsto
\sum_p \Big[\text{只取 patch 代表点的 } \varepsilon_p,\varepsilon_{p+Q}\text{ 算出的 bubble}\Big]\times(\cdots)
}
]

也就是：

* **patch 标签本身是等权的**
* 唯一的不等权来自代表点能量进入 bubble 后给出的数值差异
* 但**没有**几何 measure：

  * 没有 patch 弧长
  * 没有 (1/|v_F|)
  * 没有 patch 内平均

这点从代码上是直接成立的。 

---

# 五、这在物理上意味着什么

这意味着你现在的离散 one-loop 更像下面这个对象：

[
\Phi_{\text{code}}(Q)
\sim
\sum_p
L!\left(\varepsilon_p,\varepsilon_{\text{partner}(p,Q)}\right)
,\Gamma\Gamma
]

而不是更接近连续理论的

[
\Phi_{\text{phys}}(Q)
\sim
\sum_p
\frac{\ell_p}{|v_F(p)|}
,L_{\text{local}}(p,Q),
\Gamma\Gamma.
]

这两个东西差别很大。

---

## 1. 你的代码保留了什么

它保留了：

* transfer momentum (Q) 的几何关系；
* patch partner map；
* patch 代表点能量对 bubble 的影响；
* Matsubara 温度流结构。

所以它**不是胡算**，也不是完全没有物理内容。

---

## 2. 它丢掉了什么

它丢掉了：

* patch 在费面上覆盖多长；
* 哪些 patch 在连续积分里本来占更大 phase space；
* van Hove / 低 (v_F) 区域在费面参数化里天然更大的权重；
* patch 内部而非单点的平均。

这就是你现在最担心的那个点。

---

# 六、为什么这会系统性偏向 (Q=0)？

这一步不能说成“已经证明”，但物理逻辑非常强。

如果你的 loop 是

[
\sum_p \text{等权样本}
]

那它更容易偏向“在哪些 patch 上都能有贡献、而且贡献比较平滑”的通道。

而 (Q=0) 的特征恰恰是：

* 对很多 patch 都能自洽配对；
* partner map 往往最简单；
* bubble family 在整个矩阵上重复使用得更多。

你自己之前已经把 raw ph-(Q=0) 的这个结构看得很清楚了：
对 ph direct，整个 raw ph-(Q=0) block 的所有矩阵元都用同一个 (Q=0) internal cache，于是很容易形成 dense smooth background。

所以如果再叠加“patch 默认等权”，那它确实会更倾向把 (Q=0) 做成一个全局背景优势，而不是把 weight 更集中地放到有限个真正 nesting/van-Hove 最强的 (Q=M) patch 对上。

这和你现在看到的现象是高度一致的：

* (Q=M) 会涨；
* 但总差一点；
* (Q=0) 总像有一个普适背景。

---

# 七、因此现在可以把“第一步”总结成一句很硬的话

[
\boxed{
\text{你当前代码的 one-loop patch 离散，不是带几何 measure 的 patch 积分，}
}
]

[
\boxed{
\text{而是“每个 patch 一个代表点，用代表点能量算 bubble，再按 patch 标签求和”的近似。}
}
]

换句话说：

* 不是 patch 长度；
* 不是 patch 长度除以 (v_F)；
* 不是 patch 内 bubble 平均；
* 而是**patch-representative bubble with equal patch counting**。 

---

