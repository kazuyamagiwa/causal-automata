---
title: "グラフのセルオートマトンから、超高速な因果発見エンジンへ"
emoji: "🧬"
type: "tech"
topics: ["python", "jax", "causalinference", "githubactions", "uv", "cellularautomata"]
published: false
---

# グラフのセルオートマトンから、超高速な因果発見エンジンへ

前回の記事（[スマホとCursorだけで、6年放置したリポジトリをuv時代の構成に作り直した話](https://zenn.dev/katnap/articles/zenn-cursor-smartphone-uv-modernization)）では、[graphnet-automata](https://github.com/kazuyamagiwa/graphnet-automata) を現代化しました。あのリポジトリの核は「隣接行列を局所ルールで更新し、ネットワークを育てる」というアイデアでした。

本稿はその続きです。同じ「行列をセルオートマトン（CA）として進化させる」発想が、観測データから DAG（有向非巡回グラフ）を復元する **因果発見エンジン** に化けた経緯を、アイデア → 物理メタファー → JAX による爆速化 → パッケージ公開まで一気にたどります。

成果物はこちらです。

- GitHub: [kazuyamagiwa/causal-automata](https://github.com/kazuyamagiwa/causal-automata)
- PyPI: [causal-automata](https://pypi.org/project/causal-automata/)
- Docs: [causal-automata.readthedocs.io](https://causal-automata.readthedocs.io/)

---

## 1. 原点：グラフ上のセルオートマトン

古典的な 1D セルオートマトンでは、各セルの次状態は左右近傍のパターンで決まります。`graphnet-automata` でやっていたのは、その発想を **ネットワークの隣接行列** に持ち込むことでした。

- 状態 = 隣接行列 $A$（誰と誰がつながっているか）
- 更新 = 「近傍の接続パターン」に基づく局所ルール
- 時間発展 = $A$ を何ステップも回して、コミュニティや向きのある構造を眺める

疑似コードにすると、だいたいこんな感じです。

```python
# graphnet-automata 的な世界観（イメージ）
for t in range(steps):
    A = local_rule(A, kernel)  # 近傍パターン → エッジの生死
```

ここで大事なのは、**グラフそのものを“場”として更新している** という点です。固定したトポロジの上で情報を流すのではなく、トポロジ自体が進化する。この「隣接行列が主役」という感覚が、のちの因果発見への転回を支えます。

---

## 2. 因果へのピボット：局所ルールで DAG を掘り当てる

因果発見の古典的アプローチは、PC アルゴリズムのような **制約ベース** の探索です。独立性検定を繰り返し、エッジを削り、向きを決めていく。正しく動く一方で、候補エッジと条件集合の組み合わせはすぐ組み合わせ爆発します。

あるとき「Aha!」が来ました。

> CA の局所行列更新は、独立性／条件付き独立性の“力”を並べて書けば、そのまま DAG 探索になるのでは？

観測データから得られる相互情報量や偏相関は、エッジを生む／殺す／向き付ける **スカラーな力** として読めます。隣接行列 $A$ を時間発展させ、最終的に残った矢印を因果構造とみなす——これは PC の逐次探索とはまったく別の経路です。

| 観点 | PC など制約ベース | Causal Automata |
| --- | --- | --- |
| 探索の単位 | エッジ単位の逐次検定 | 行列全体の並列更新 |
| 状態 | CPDAG / スケルトン | 隣接行列 $A$ + 標高 $E$ |
| 向き付け | 制約ルールの連鎖 | 物理メタファー（重力・シンク） |
| 実装の相性 | 分岐だらけの制御フロー | テンソル演算・JIT |

「独立性検定の結果をルール表に落とす」のではなく、「検定統計量を物理量として場に流し込む」。この言い換えが、のちの GPU 化をほぼ必然にします。

---

## 3. 因果オートマトンの「物理」

### Forks vs Colliders：因果発見の核心課題

観測データだけでは相関しか見えません。アイスクリームの売上とサメの襲撃は夏に同時に増えますが、因果ではありません。ここに現れるのが **フォーク（共通原因）** と **コライダー（V 構造）** の違いです。

```text
# Fork（共通原因）— 条件付けで依存が消える
      Temperature
       /        \
      v          v
  Ice cream   Shark attacks

# Collider / V-structure — 条件付けで依存が開く
  Rain          Sprinkler
    \             /
     v           v
      Grass wet
```

- フォークでは、共通原因を知ると子ども同士が条件付き独立になる
- コライダーでは、シンクを観測すると親同士が依存し合う

PC はこのパターンを制約として扱い分けます。われわれはそれを **力** としてモデル化しました。

### Gravity（重力）：相関がエッジを引き寄せる

各ノードに標高 $E \in [0,1]$ を持たせ、「エッジは高いところから低いところへしか流れない」と決めます。これだけでサイクルが物理的に禁止されます（下り坂しかないから）。

さらに、親・子の平均標高の中点へ $E$ をドリフトさせます。強い結合があるノード同士が標高を寄せ合い、向きの一貫性が育ちます。これが **Gravity** です。

```python
# 下り坂マスクでサイクルを禁止し、標高を中点へ引き寄せる
gravity_ok = E[:, None] > E[None, :]          # parent が高いときだけ
A = A * gravity_ok

mean_parents = (A.T @ E) / jnp.maximum(A.sum(0), 1)
mean_children = (A @ E) / jnp.maximum(A.sum(1), 1)
target = 0.5 * (mean_parents + mean_children)
E = jnp.clip(E + lr * (target - E), 0.0, 1.0)
```

### V-Well / Sink：コライダーを沈めて向きを固定する

条件付き相互情報量（線形ガウスなら偏相関で近似）を使い、「縁のない二点 $i,j$ が共有近傍 $k$ を持ち、かつ周辺独立」というパターンを検出します。その共有近傍を **コライダー候補** として標高を沈める——これが **V-Well（シンク）** です。

標高が下がると、次の birth ステップでは矢印が $i \to k \leftarrow j$ に向きやすくなります。サイクルを壊しつつ、V 構造を地形として刻み込むイメージです。

```python
# 共有近傍 × 周辺独立 → コライダーを沈める
U = ((A + A.T) > 0).astype(A.dtype)           # 無向近傍
shared = (U @ U) > 0
independent = jnp.abs(rho_empty) < thr_death  # 空の条件集合での独立性
collider = ...                                # shared ∩ independent の近傍
E = jnp.where(collider, E - 2.0 * lr, E)
```

1 ティックの流れをまとめるとこうなります。

1. **Gravity** — 下り坂エッジだけ残し、標高をドリフト
2. **V-Well** — コライダー候補を沈める
3. **Death** — 偏相関が弱い既存エッジを切る
4. **Birth** — 偏相関が強い未接続ペアを、標高に従って向き付けて生やす

相互情報量／偏相関は、ここでは「統計量」であると同時に「場にかかる力」です。

---

## 4. JAX の突破口とベンチマーク

初期実装は NumPy のブール／疎なループに近い発想でした。因果発見に振り切った瞬間、ボトルネックは明らかでした。**全ペアの偏相関を、毎ティック、条件集合つきで計算する** 必要がある。

そこで状態を連続テンソルに寄せ、演算をすべて `jax.numpy` のバッチ線形代数に書き換えました。`step` 自体は `@jax.jit` でコンパイルし、ティック列は `jax.lax.scan` で Python ループを消します。

```python
import jax
import jax.numpy as jnp
from causal_automata.jax_core import step

@jax.jit
def run(A0, E0, Cov, ticks: int = 100, k: int = 3):
    def body(carry, _):
        A, E = carry
        A, E = step(A, E, Cov, k=k)  # gravity → V-well → death → birth
        return (A, E), None

    (A, E), _ = jax.lax.scan(body, (A0, E0), xs=None, length=ticks)
    return A, E
```

ポイントは次のとおりです。

- 隣接行列・標高・共分散はすべて固定形状のテンソル
- 条件集合サイズ `k` は JIT の static 引数（`(2+k)×(2+k)` のバッチ逆行列が専門化される）
- `scan` により「100 ティック」が一つの XLA プログラムになる

### 数字で見るジャンプ

合成線形 DAG（ノード数 $N=100$）を Colab の **Tesla T4** で回したときの代表値です（歴史的設定 `k=3`）。

| アルゴリズム | Skeleton F1 | Directed F1 | 壁時計時間 |
| --- | ---: | ---: | ---: |
| PC（gCastle） | 0.968 | 0.813 | **528.631 s** |
| GOLEM（gCastle） | 0.577 | 0.403 | 251.182 s |
| Bayesian CA（JAX） | 0.812 | 0.483 | **0.323 s** |

PC に対しておよそ **1,600 倍**（$528 / 0.3 \approx 1{,}600$）です。精度で PC に勝つことが目的ではなく、「同じクラスの合成ハーネスで、**秒未満**で骨格と向きが立ち上がる」ことが驚きでした。

現行のデフォルト（$N>50$ では `k=5`）では Directed F1 が上がり、壁時計は CPU JAX でおおよそ 3 秒前後になります。さらに `--multihop`（$A@A$ による 2-hop Gravity）を入れると、同じ時間帯のまま Directed F1 が **0.696** まで伸びます。速さの物語の象徴として、最初に体感した **0.3 秒** は今も強く残っています。

```bash
# 手元で再現するなら
pip install "causal-automata[jax,gcastle]"
python -m causal_automata.benchmark --n 100 --seed 0 --ticks 100
```

---

## 5. スピンアウトと近代化

アイデアが「研究用ノート」から「使えるエンジン」に見えてきたタイミングで、graphnet-automata から切り離して専用リポジトリ **causal-automata** を切り出しました。前回の近代化作業の延長線上で、最初から現代的な Python パッケージとして組み立てています。

### `pyproject.toml` + `uv` + ハードウェア extras

コア依存は CPU でも軽い NumPy / NetworkX / scikit-learn に留め、アクセラレータは extras に分離しました。

```toml
[project.optional-dependencies]
# CPU 向け JAX
jax = [
    "jax>=0.4.30",
    "jaxlib>=0.4.30",
]
# NVIDIA / Colab 向け GPU スタック
cuda = [
    "jax[cuda13]",
    "torch>=2.1",
    "gcastle>=1.0.4",
]
```

使い分けはこうです。

```bash
# CPU
uv pip install "causal-automata[jax]"

# GPU（CUDA）
uv pip install "causal-automata[cuda]"
```

「まず動かしたい人」と「ベンチマークや Colab GPU で本気を出したい人」を、同じパッケージで分岐させられます。

### GitHub Actions × Trusted Publishing（OIDC）

公開は GitHub Releases をトリガに、`uv build` → `uv publish` を走らせています。PyPI の **Trusted Publishing** により、API トークンをリポジトリに置かず、OIDC で一時クレデンシャルを発行します。

```yaml
# .github/workflows/publish.yml（抜粋）
on:
  release:
    types: [published]

permissions:
  id-token: write   # OIDC
  contents: read

steps:
  - uses: actions/checkout@v4
  - uses: astral-sh/setup-uv@v5
  - run: uv build
  - run: uv publish
```

Release を切るだけで PyPI に載る、という状態まで自動化できたのは、前回 graphnet-automata を uv 化したときの延長で本当に気持ちよかったです。

---

## まとめ

物語を短くすると、こうです。

1. **原点** — 隣接行列を CA で育てる（graphnet-automata）
2. **転回** — 局所更新ルールを、観測データからの DAG 発見に読み替える
3. **物理** — Gravity と V-Well でフォーク／コライダーを地形として扱う
4. **加速** — ブール行列から連続テンソルへ。`@jit` + `lax.scan` で GPU 上 0.3 秒、PC 比おおよそ 1,600 倍
5. **公開** — `causal-automata` として切り出し、`uv` extras と OIDC で PyPI へ

セルオートマトンの「局所ルールで場が変わる」感覚は、因果発見の「独立性のパターンで矢印が立つ」感覚と、思ったより近くにありました。次に触る人が同じ興奮を共有できるように、コードもドキュメントもパッケージもそろえてあります。

### リンク

- 前回記事（graphnet-automata の近代化）: https://zenn.dev/katnap/articles/zenn-cursor-smartphone-uv-modernization
- 原点リポジトリ: https://github.com/kazuyamagiwa/graphnet-automata
- 本編リポジトリ: https://github.com/kazuyamagiwa/causal-automata
- PyPI: https://pypi.org/project/causal-automata/
- Read the Docs: https://causal-automata.readthedocs.io/
