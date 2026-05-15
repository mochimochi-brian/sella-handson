# Sella クイックスタート

[Sella](https://github.com/zadorlab/sella) は、原子系の **鞍点最適化 (saddle point optimization)** と
**局所最小化 (minimization)** を行うための Python ライブラリです。[ASE (Atomic Simulation
Environment)](https://wiki.fysik.dtu.dk/ase/) と統合されており、NWChem, Quantum Espresso, VASP など
20 種類以上の電子状態計算パッケージから力 (forces) を受け取って、遷移状態 (TS) 探索や IRC 計算を
実行できます。

このハンズオンでは、EMT (Effective Medium Theory) の安価な力場を使って、Sella の使い方を一通り
体験します。EMT は ASE に同梱されているため、追加のセットアップなしで動きます。

---

## 1. インストール

Python 3.8 以上が必要です。仮想環境を作って `pip` でインストールします。

```bash
python -m venv .venv
source .venv/bin/activate
pip install sella
```

`sella` を入れると、依存関係として `ase`, `numpy`, `scipy`, `jax`, `jaxlib` も自動でインストール
されます。動作確認:

```bash
python -c "import sella, ase; print(sella.__version__, ase.__version__)"
```

---

## 2. はじめての鞍点探索 (Cu(111) 上の Cu 吸着原子)

`examples/01_saddle_point.py` として保存して実行してください。

```python
#!/usr/bin/env python3
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT

from sella import Sella, Constraints

# 1. ASE Atoms オブジェクトとして系を組み立てる
slab = fcc111('Cu', (5, 5, 6), vacuum=7.5)
add_adsorbate(slab, 'Cu', 2.0, 'bridge')

# 2. 拘束 (constraints) を設定: スラブの下半分は動かさない
cons = Constraints(slab)
for atom in slab:
    if atom.position[2] < slab.cell[2, 2] / 2.0:
        cons.fix_translation(atom.index)

# 3. 力 (forces) を計算する calculator を紐付ける
slab.calc = EMT()

# 4. Sella を Dynamics オブジェクトとしてセットアップ
dyn = Sella(
    slab,
    constraints=cons,
    trajectory='cu_saddle.traj',
)

# 5. 収束判定 fmax=1e-3 eV/Å, 最大 1000 ステップで実行
dyn.run(1e-3, 1000)
```

実行後:

```bash
ase gui cu_saddle.traj         # 軌道を可視化
python -c "from ase.io import read; a = read('cu_saddle.traj'); print(a.get_potential_energy())"
```

`Sella` クラスは ASE の `Optimizer` 互換なので、`BFGS` や `FIRE` と同じ感覚で差し替えられます。

---

## 3. 最小化モードに切り替える

デフォルトでは 1 次鞍点を探しますが、`order=0` を渡せば通常の最小化になります。
TS 探索の前に初期構造を緩和したい場合に使います。

```python
dyn = Sella(slab, constraints=cons, order=0, trajectory='cu_min.traj')
dyn.run(1e-3, 1000)
```

| `order` | 探索対象              |
|---------|-----------------------|
| 0       | 局所最小 (minimum)    |
| 1       | 1 次鞍点 (TS)         |
| 2       | 2 次鞍点              |

---

## 4. 拘束 (Constraints) の主な書き方

```python
cons = Constraints(slab)

# 並進: 原子 0 を xyz 全方向で固定
cons.fix_translation(0)

# 並進: 原子 1 を x 方向のみ固定
cons.fix_translation(1, dim=0)

# 結合長: 原子 0-1 の距離を現在値で固定
cons.fix_bond((0, 1))

# 結合長: 原子 1-2 の距離を 1.5 Å に固定 (反応座標を保持したい時など)
cons.fix_bond((1, 2), target=1.5)

# 結合角: 原子 1-2-3 の角度を 120° に固定
cons.fix_angle((1, 2, 3), target=120)

# 二面角: 原子 0-1-2-3 を現在値で固定
cons.fix_dihedral((0, 1, 2, 3))
```

周期境界をまたぐ結合は `mic=True` か `ncvecs=...` で指定します。

---

## 5. ハイパーパラメータの調整

Sella は内部で Hessian の部分対角化と trust radius 更新を行います。
デフォルトでも大抵動きますが、収束しにくい時はここを調整します。

| 引数        | 役割                                                       | 既定値      |
|-------------|------------------------------------------------------------|-------------|
| `eta`       | 有限差分のステップ幅 (Å)                                   | `1e-4`      |
| `method`    | 対角化アルゴリズム (`jd0` / `gd` / `lanczos`)              | `jd0`       |
| `gamma`     | 固有値ソルバの収束基準                                     | `0.4` 付近  |
| `delta0`    | 自由度あたりの初期 trust radius (Å)                        | `0.1` 付近  |
| `rho_inc`   | trust radius を縮める閾値                                  | -           |
| `rho_dec`   | trust radius を広げる閾値                                  | -           |
| `sigma_inc` | trust radius を広げる倍率                                  | -           |
| `sigma_dec` | trust radius を縮める倍率                                  | -           |

例: 力のノイズが大きい計算で `eta` を緩める。

```python
dyn = Sella(slab, eta=5e-4, gamma=0.2, trajectory='tuned.traj')
```

---

## 6. IRC (固有反応座標) を流す

鞍点が収束したら、そこから前向き／後ろ向きに転がして反応経路を確認します。

```python
from ase.io import read
from sella import IRC

atoms = read('cu_saddle.traj@-1')   # 最終フレーム = 収束した鞍点
atoms.calc = EMT()

irc = IRC(atoms, trajectory='irc.traj', dx=0.1, eta=1e-4, gamma=0.4)
irc.run(fmax=0.1, steps=1000, direction='forward')
irc.run(fmax=0.1, steps=1000, direction='reverse')
```

- `dx`: IRC イメージ間隔 (Å·√amu 単位)
- `direction`: `'forward'` または `'reverse'`
- `keep_going=True`: 内部反復が失敗してもループを継続

---

## 7. よくあるつまずき

- **`order=1` のはずなのに最小化に落ちる** — 初期構造が鞍点に近くないと、Hessian の負固有値方向を
  捕まえられず最小に滑り落ちます。NEB などで近い構造を作ってから渡してください。
- **力が振動して収束しない** — `eta` を大きく (`5e-4` ～ `1e-3`)、`gamma` を小さく (`0.2` 程度) して
  みる。`delta0` を 0.05 まで絞るのも有効です。
- **拘束した結合が動いてしまう** — `Constraints` オブジェクトを `Sella(..., constraints=cons)` で
  渡し忘れているケースが多いです。`atoms.constraints` に直接セットしても効きません。
- **JAX のインストールで失敗する** — `pip install --upgrade pip` してから再試行。GPU 版 jaxlib は
  不要です (CPU 版で十分)。

---

## 8. 参考リンク

- [zadorlab/sella (GitHub)](https://github.com/zadorlab/sella)
- [Sella Wiki (Constraints / Hyperparameters / IRC)](https://github.com/zadorlab/sella/wiki)
- [PyPI: sella](https://pypi.org/project/sella/)
- [論文: Sella, an Open-Source Automation-Friendly Molecular Saddle Point Optimizer (JCTC 2022)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00395)
- [ASE 公式ドキュメント](https://wiki.fysik.dtu.dk/ase/)
