# Order Diagnosis — cells to append to `pipeline.ipynb`

下面这部分是**直接接在你当前 `pipeline.ipynb` 的 `# Order Diagnosis` 后面**的 notebook 结构。
我刻意沿用了你前面 notebook 已经存在的变量名：`model`, `patchsets`, `interaction`, `decomp`。
因此你不需要重构前四个模块，只需要把这些 cell 追加进去。

---

## Cell 1 — Markdown

```markdown
# Order Diagnosis

这一部分不再测试 channel decomposition 本身，而是把前面已经构造好的离散 channel kernel 当成“候选有序核”，对不同的转移动量 $Q$ 和不同的自旋/配对结构做本征分解。

我们要检查的主要对象是：

- **$Q=0$ particle-hole spin channel**：对应 ferromagnetism (FM) 的候选；
- **$Q=0$ particle-hole charge channel**：对应 Pomeranchuk instability (PI) 的候选；
- **$Q=0$ particle-particle triplet channel**：对应 triplet / $f$-wave superconductivity 的候选；
- **$Q=Q_1,Q_2,Q_3$ particle-hole charge/spin channel**：对应 charge bond order (cBO) / spin bond order (sBO) 的候选。

这里的目标还不是最终“严格证明”它一定是哪一种 order，而是先完成：

1. 固定候选 channel；
2. 取该 kernel 的 leading eigenmode；
3. 观察其在 patch 空间中的振幅与符号结构；
4. 为后续 real-space / bond-form-factor 诊断提供输入。
```

---

## Cell 2 — Code

```python
# Candidate transfer momenta used in the Kagome-Hubbard paper
# Lattice constant between adjacent sites is set to 1.

Q0 = np.array([0.0, 0.0])
Q1 = np.array([-0.5, -np.sqrt(3)/2])
Q2 = np.array([ 1.0,  0.0])
Q3 = np.array([-0.5,  np.sqrt(3)/2])

Q_candidates = {
    "Q0": Q0,
    "Q1": Q1,
    "Q2": Q2,
    "Q3": Q3,
}

for name, Q in Q_candidates.items():
    print(f"{name} = {Q}")
```

---

## Cell 3 — Code

```python
from dataclasses import dataclass

@dataclass
class LeadingMode:
    channel_name: str
    Q_name: str
    Q: np.ndarray
    eigenvalue: complex
    eigenvector: np.ndarray
    kernel: object


def fix_mode_phase(vec: np.ndarray) -> np.ndarray:
    """
    Fix a global U(1) phase for easier comparison between eigenmodes.
    Strategy: make the largest-amplitude component real and positive.
    """
    v = np.asarray(vec, dtype=complex).copy()
    i0 = int(np.argmax(np.abs(v)))
    if np.abs(v[i0]) < 1e-14:
        return v
    phase = np.angle(v[i0])
    v *= np.exp(-1j * phase)
    if np.real(v[i0]) < 0:
        v *= -1
    return v


def leading_mode(kernel, *, sort_by="abs"):
    vals, vecs = kernel.eig(sort_by=sort_by)
    vec0 = fix_mode_phase(vecs[:, 0])
    return vals[0], vec0


def print_mode_summary(name, kernel, *, topn=6, sort_by="abs"):
    val, vec = leading_mode(kernel, sort_by=sort_by)
    amp = np.abs(vec)
    order = np.argsort(-amp)

    print("=" * 80)
    print(f"channel = {name}")
    print(f"Q       = {kernel.Q}")
    print(f"shape   = {kernel.matrix.shape}")
    print(f"lambda0 = {val}")
    print(f"|lambda0| = {abs(val)}")
    print(f"hermitian residual = {kernel.hermitian_residual():.6e}")
    print(f"max patch-match residual = {np.max(kernel.residuals):.6e}")
    print("top components:")
    for j in order[:topn]:
        print(
            f"  patch {j:3d}: "
            f"vec = {vec[j]: .6f}, "
            f"|vec| = {abs(vec[j]):.6f}"
        )
```

---

## Cell 4 — Code

```python
def plot_mode_on_patchset(patchset, vec, *, ax=None, title="", annotate=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ks = np.array([p.k_cart for p in patchset.patches], dtype=float)
    vec = fix_mode_phase(vec)
    amp = np.abs(vec)
    phase = np.angle(vec)

    sc = ax.scatter(ks[:, 0], ks[:, 1], c=phase, s=250 * (amp / amp.max() + 0.08), cmap="twilight")
    ax.axhline(0.0, lw=0.8, c="gray", alpha=0.5)
    ax.axvline(0.0, lw=0.8, c="gray", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="phase(arg eigenvector)")

    if annotate:
        for i, (x, y) in enumerate(ks):
            ax.text(x, y, str(i), fontsize=8, ha="center", va="center")

    return ax


def plot_mode_components(vec, *, ax=None, title=""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    vec = fix_mode_phase(vec)
    idx = np.arange(len(vec))
    ax.plot(idx, np.real(vec), marker='o', label='Re')
    ax.plot(idx, np.imag(vec), marker='s', label='Im')
    ax.plot(idx, np.abs(vec), marker='^', label='|.|')
    ax.set_xlabel("patch index")
    ax.set_ylabel("mode component")
    ax.set_title(title)
    ax.legend()
    return ax
```

---

## Cell 5 — Code

```python
def build_candidate_kernels(decomp, Q_dict):
    out = {}

    for qname, Q in Q_dict.items():
        # particle-hole longitudinal charge/spin
        ph = decomp.ph_charge_spin_longitudinal(Q)
        out[(qname, "ph_charge")] = ph["charge"]
        out[(qname, "ph_spin")]   = ph["spin"]

        # particle-particle singlet/triplet
        pp = decomp.pp_singlet_triplet_sz0(Q)
        out[(qname, "pp_singlet_sz0")] = pp["singlet_sz0"]
        out[(qname, "pp_triplet_sz0")] = pp["triplet_sz0"]

    return out


candidate_kernels = build_candidate_kernels(decomp, Q_candidates)
print(f"Built {len(candidate_kernels)} candidate kernels.")
print(sorted(candidate_kernels.keys()))
```

---

## Cell 6 — Code

```python
results = []
leading_modes = {}

for (qname, cname), kernel in candidate_kernels.items():
    lam, vec = leading_mode(kernel, sort_by="abs")
    leading_modes[(qname, cname)] = LeadingMode(
        channel_name=cname,
        Q_name=qname,
        Q=np.asarray(kernel.Q, dtype=float),
        eigenvalue=lam,
        eigenvector=vec,
        kernel=kernel,
    )
    results.append({
        "Q_name": qname,
        "channel": cname,
        "eigenvalue": lam,
        "abs_eigenvalue": abs(lam),
        "max_patch_match_residual": float(np.max(kernel.residuals)),
        "hermitian_residual": float(kernel.hermitian_residual()),
    })

results = sorted(results, key=lambda x: x["abs_eigenvalue"], reverse=True)

for row in results:
    print(
        f"Q={row['Q_name']:>2s} | "
        f"channel={row['channel']:<16s} | "
        f"lambda0={row['eigenvalue']} | "
        f"|lambda0|={row['abs_eigenvalue']:.6f}"
    )
```

---

## Cell 7 — Code

```python
fig, ax = plt.subplots(figsize=(10, 5))
labels = [f"{r['Q_name']} / {r['channel']}" for r in results]
vals = [r["abs_eigenvalue"] for r in results]
ax.bar(np.arange(len(vals)), vals)
ax.set_xticks(np.arange(len(vals)))
ax.set_xticklabels(labels, rotation=60, ha='right')
ax.set_ylabel(r"leading $|\lambda|$")
ax.set_title("Candidate order channels ranked by leading eigenvalue magnitude")
plt.tight_layout()
```

---

## Cell 8 — Markdown

```markdown
## Readout guide

粗略上可以按下面的方式读这个 ranking：

- 若 **`Q0 / ph_spin`** 最强，优先联想到 **FM**；
- 若 **`Q0 / ph_charge`** 最强，优先联想到 **PI / nematic`**；
- 若 **`Q1,Q2,Q3 / ph_charge`** 最强，优先联想到 **cBO**；
- 若 **`Q1,Q2,Q3 / ph_spin`** 最强，优先联想到 **sBO**；
- 若 **`Q0 / pp_triplet_sz0`** 最强，优先联想到 **triplet superconductivity**；
- 若其 patch-space eigenvector 呈现明显角动量换号结构，再进一步检查是否和 **$f$-wave** 一致。

注意：这里的 notebook 还只是 **patch-space diagnosis**，不是最终的 real-space bond pattern reconstruction。
最终要和论文中的 cBO / sBO / PI / f-SC 一一对应，下一步还需要把 leading eigenvector 再映射成 form factor 或 bond pattern。
```

---

## Cell 9 — Code

```python
# Inspect the top few candidates in detail
for row in results[:4]:
    qname = row["Q_name"]
    cname = row["channel"]
    mode = leading_modes[(qname, cname)]
    print_mode_summary(f"{qname} / {cname}", mode.kernel, topn=8)
```

---

## Cell 10 — Code

```python
# Visualize the top candidate on the patch set
best = results[0]
mode = leading_modes[(best["Q_name"], best["channel"])]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_mode_on_patchset(
    patchsets["up"],
    mode.eigenvector,
    ax=axes[0],
    title=f"Leading mode on patches\n{best['Q_name']} / {best['channel']}",
)
plot_mode_components(
    mode.eigenvector,
    ax=axes[1],
    title=f"Patch components\n{best['Q_name']} / {best['channel']}",
)
plt.tight_layout()
```

---

## Cell 11 — Code

```python
# Side-by-side comparison of the physically most relevant candidates
focus_list = [
    ("Q0", "ph_spin"),         # FM candidate
    ("Q0", "ph_charge"),       # PI candidate
    ("Q0", "pp_triplet_sz0"),  # f-SC candidate at Q=0
    ("Q1", "ph_charge"),       # cBO candidate
    ("Q1", "ph_spin"),         # sBO candidate
]

fig, axes = plt.subplots(len(focus_list), 2, figsize=(12, 4 * len(focus_list)))

for i, key in enumerate(focus_list):
    mode = leading_modes[key]
    plot_mode_on_patchset(
        patchsets["up"],
        mode.eigenvector,
        ax=axes[i, 0],
        title=f"{key[0]} / {key[1]}"
    )
    plot_mode_components(
        mode.eigenvector,
        ax=axes[i, 1],
        title=f"components: {key[0]} / {key[1]}"
    )

plt.tight_layout()
```

---

## Cell 12 — Code

```python
# Compare symmetry-related finite-Q bond-order candidates
for cname in ["ph_charge", "ph_spin"]:
    print("\n" + "#" * 80)
    print(f"Comparing {cname} among Q1,Q2,Q3")
    for qname in ["Q1", "Q2", "Q3"]:
        mode = leading_modes[(qname, cname)]
        print(f"{qname}: lambda0 = {mode.eigenvalue}, |lambda0| = {abs(mode.eigenvalue):.6f}")
```

---

## Cell 13 — Markdown

```markdown
## What to look for after running

运行完以后，你把 notebook 发给我，我会重点帮你看三件事：

1. **哪个 channel 真正最大**：它决定你当前 bare setup 更偏向哪类 instability；
2. **Q1/Q2/Q3 是否近简并**：这和 kagome 上 cBO / sBO 三个方向同时形成有关；
3. **leading mode 的换号结构**：这决定它更像 FM / PI / bond order / triplet pairing 中的哪一种。

如果这一轮结果合理，下一步就该做的是：

- 把 leading eigenvector 进一步写成更明确的 **form factor basis projection**；
- 或者直接构造 **real-space bond expectation pattern**；
- 再去和论文图里的 cBO / sBO / PI / f-SC 图样一一对照。
```

---

## Optional Cell 14 — Code

```python
# Save a lightweight text summary for later comparison with FRG-flow outputs
summary_lines = []
summary_lines.append("Order diagnosis summary\n")
summary_lines.append("=" * 80 + "\n")

for row in results:
    summary_lines.append(
        f"Q={row['Q_name']:<2s} | "
        f"channel={row['channel']:<16s} | "
        f"lambda0={row['eigenvalue']} | "
        f"|lambda0|={row['abs_eigenvalue']:.8f} | "
        f"match_res={row['max_patch_match_residual']:.3e} | "
        f"herm_res={row['hermitian_residual']:.3e}\n"
    )

with open("order_diagnosis_summary.txt", "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

print("Saved: order_diagnosis_summary.txt")
```
