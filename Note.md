# Artificial Intelligence

## 知识表示和推理

### 前置知识

1. 项：常量、函数和参数的复合。eg: a，f(a)，f(x)

2. ==**置换**==：置换是形如 $\{x_1/t_1, x_2/t_2, \ldots, x_n/t_n\}$ 的有限集合，其中：

   - $\{x_1, x_2, \ldots, x_n\}$ 是互不相同的变量

   - $\{t_1, t_2, \ldots, t_n\}$ 是项（常量、变量或函数）

   - $x_i/t_i$ 表示用 $t_i$ 置换 $x_i$，其==**合法的前提**==是：

     1. 不允许 $t_i$ 和 $x_i$ 相同

     2. 不允许 $x_i$ 出现在 $t_i$ 中，

     3. ==也不允许出现循环置换==

        如：$\{x /y \} \text{和} \{x/f(y), y/{f(x)}\}$

   注意：置换中的每个变量替换是同时进行的

   > 对于 $P(x, g(y, z))$，做置换 $\{x/y, y/f(a)\}$，
   >
   > 即 $P(x, g(y, z)) \{x/y, y/f(a)\} \Rightarrow P(y, g(f(a), z))$

3. 文字：原子公式及其否定

   - $ P $：正文字；$ \neg P $：负文字

   子句：任何文字的析取。某个文字本身也都是子句。

   - $ P \vee \neg Q $ 记作 $ (P, \neg Q) $
   - 空子句：不包含任何文字的子句，记作 NIL
     - 空子句是永假的，不可满足的。

   子句集：由子句构成的集合（子句的合取）

   - $ (P \vee \neg Q) \wedge (P \vee R) $ 记作 $ \{(P, \neg Q), (P, R)\} $

****

特殊符号

1. `⊢*Am`  或者 ``⊢Am`` ：

   公式序列 A1, A2, ..., Am 称作Am的一个证明，如果 Ai (1 ≤ i ≤ m)：
   - 或者是公理；
   - 或者由Aj1, ..., Ajk (j1, ..., jk < i)用推理规则推得。

   当这样的证明存在时，称Am为系统的定理，记作 `|-*Am`（*是形式系统的名称），或者简记为 |-Am

2. `Γ⊢*Am` 或者 `Γ⊢Am`：

   设 Γ 为一公式集合。公式序列 A1, A2, ..., Am 称作 Am 的以 Γ 为前提的演绎，如果 Ai (1 ≤ i ≤ m)：

   - 或者是 Γ 中的公式
   - 或者是公理
   - 或者由 Aj1, ..., Ajk (j1, ..., jk < i) 用推理规则推得。

   当有这样的演绎时，Am 称作 Γ 的演绎结果，记作 `Γ⊢*Am`（*是形式系统的名称），或者简记为 `Γ⊢Am`，称 Γ 和 Γ 的成员为 Am 的前提

3. $\models_i$ 或者 $ \models $：

   如果推理算法 $ i $ 可以根据 $ KB $ 导出结论 $ \alpha $，则形式化地记为：$ KB \models_i \alpha $

   将 $ S $ 逻辑上蕴含 $ C $ 记为 $ S \models C $

4. $\vdash$：

   记某个永真的子句集合为 $ S $，需要推理得到的子句为 $ C $，基于归结的推理过程从 $ S $ 推导出 $ C $ 记为 $ S \vdash C $

### 归结推理

1. 归结式：对于任意两个子句 $C_1$ 和 $C_2$，若 $C_1$ 中有一个文字 $L$，而 $C_2$ 中有一个与 $L$ 成互补的文字 $\neg L$，则分别从 $C_1$ 和 $C_2$ 中删去 $L$ 和 $\neg L$，并将其剩余部分组成新的析取式。这个新的子句被称为 $C_1$ 和 $C_2$ 关于 $L$ 的归结式，$C_1$ 和 $C_2$ 则是该归结式的亲本子句。

   * 子句 $P$ 和 $\neg P$ 的归结式为空子句
   * 子句 $(W, R, Q)$ 和 $(W, S, \neg R)$ 的归结式为 $(W, Q, S)$

   **定理**：两个子句的归结式是这两个子句集的逻辑推论，如 $\{(P, C_1), (\neg P, C_2)\} \models (C_1, C_2)$

2. 如果 $ S \vdash C $，那么 $ S \models C $

   如果 $ S \vdash NIL $，那么 $ S \models NIL $，反之亦然

3. 鲁滨逊归结原理：检查子句集 S 中是否包含空子句，若包含，则 S 不可满足；若不包含，则在 S 中选择合适的子句进行归结，一旦归结出空子句，就说明 S 是不可满足的

5. 合一：在谓词逻辑的归结过程中，寻找项之间合适的变量置换使表达式一致，这个过程称为合一。

   * 用 $ \sigma = \{x_1/t_1, x_2/t_2, \ldots, x_n/t_n\} $ 来表示任一置换。用 $ \sigma $ 对表达式（语句）$ S $ 作置换后的例简记为 $ S\sigma $。

   * 可以对表达式多次置换：如用 $ \theta $ 和 $ \sigma $ 依次对 $ S $ 进行置换，记为 $ (S\theta)\sigma $。其结果等价于先将这两个置换合成（组合）为一个置换，即 $ \theta\sigma $，再用合成置换对 $ S $ 进行置换，即 $ S(\theta\sigma) $

6. 置换复合的过程：

   设 $ \theta = \{x_1/t_1, x_2/t_2, \ldots, x_n/t_n\} $，$ \sigma = \{y_1/u_1, y_2/u_2, \ldots, y_n/u_n\} $

   1. 构成 $\{x_1/t_1 \sigma, \ldots, x_n/ t_n\sigma, y_1/u_1, \ldots, y_m/u_m\}$；
   2. 如果 $y_j \in (x_1, \ldots, x_n)$，则删除 $y_j/u_j$；
   3. 如果 $t_k \sigma = x_k$，则删除 $x_k / t_k \sigma$;

   > 置换的合成公式比较复杂，不妨看个例子
   >
   > 令 $\theta = \{x / f(y), y / z\}, \sigma = \{x / a, y / b, z / y\}$
   >
   > 步骤1：$\theta \sigma = \{x / f(b), y / y, x / a, y / b, z / y\}$
   >
   > 步骤2：删除 $x / a$ 和 $y / b$
   >
   > 步骤3：删除 $y / y$
   >
   > $\theta \sigma = \{x / f(b), z / y\}$

7. 合一项：对于两个语句 $ f $ 和 $ g $，合一项是使得语句 $ f $ 和 $ g $ 等价的一个置换 $ \sigma $。

   最一般合一项：两个语句 $ f $ 和 $ g $ 的最一般合一项 $ \sigma $ 满足：

   - $ \sigma $ 是 $ f $ 和 $ g $ 的一个合一项
   - 对于 $ f $ 和 $ g $ 的任意其它合一项 $ \theta $，存在一个替换 $ \lambda $ 使得 $ \theta = \sigma \lambda $

8. 求最一般合一项：

   给定两个语句 $ f $ 和 $ g $，

   1. 初始化：$ \sigma = \{\}, S = \{f, g\} $
   2. 如果 $ S $ 包含相同的语句，那么停止算法：当前的置换 $ \sigma $ 为语句 $ f $ 和 $ g $ 的最一般合一项目
   3. 否则，找出 $ S $ 的差异集 $ D = \{e_1, e_2\} $：
      - 若 $ e_1 = v $ 是一个变量且 $ e_2 = t $ 是一个不包含 $ v $ 的项，那么令 $ \sigma = \sigma \cup \{v/t\} $，$ S = S \{v/t\} $。返回步骤 2
      - 否则，停止算法：语句 $ f $ 和 $ g $ 不可合一

9. 谓词公式化为子句集的步骤：

   以将下列谓词公式化为子句集为例：$\forall x \Big( \forall y P(x, y) \rightarrow \neg \forall y \big(Q(x, y) \rightarrow R(x, y)\big) \Big)$

   1. 消去谓词公式中的 “→” 和 “↔”
      $$
      \forall x \Big( \neg \forall y P(x, y) \vee \neg \forall y \big( \neg Q(x, y) \vee R(x, y) \big) \Big) \tag{1}
      $$

   2. 把否定符号移到紧靠谓词的位置上，减少否定符号的辖域。
      $$
      \forall x \Big( \exists y \neg P(x, y) \vee \exists y \big( Q(x, y) \wedge \neg R(x, y) \big) \Big) \tag{2}
      $$

   3. 变量标准化：重新命名变元，使每个量词采用不同的变元，从而使不同量词的约束变元有不同的名字。
      $$
      \forall x \Big( \exists y \neg P(x, y) \vee \exists z \big( Q(x, z) \wedge \neg R(x, z) \big) \Big) \tag{3}
      $$

   4. 消去存在量词

      分两种情况：
      $$
      \begin{cases} 
      \exists x \forall y \Big(\neg P(x, z) \vee R \big(x, y, f(a)\big) \Big) \Rightarrow \forall y \Big(\neg P \big(b, g(y) \big) \vee R\big(b, y, f(a)\big)\Big) \\[2ex]
      \forall x_1 \forall x_2 \ldots \forall x_n \exists y P(x_1, x_2, \ldots, x_n, y) \Rightarrow \forall x_1 \forall x_2 \ldots \forall x_n  P(x_1, x_2, \ldots, x_n, f(x_1, x_2, \dots, x_n))
      \end{cases}
      $$
      原式化为：
      $$
      \forall x \bigg( \neg P\big(x, f(x) \big) \vee \Big(Q\big(x, g(x)\big) \wedge \neg R\big(x, g(x)\big) \Big) \bigg) \tag{4}
      $$
      
   5. 化为前束范式
   
   6. 化为合取范式
      $$
      \forall x \bigg( \Big(\neg P\big(x, f(x)\big) \vee Q\big(x, g(x)\big)\Big) \wedge \Big(\neg P\big(x, f(x)\big) \vee \neg R\big(x, g(x)\big)\Big) \bigg) \tag{5}
      $$
   
   7. 略去全称量词
      $$
      \Big(\neg P\big(x, f(x)\big) \vee Q\big(x, g(x)\big)\Big) \wedge \Big(\neg P\big(x, f(x)\big) \vee \neg R\big(x, g(x)\big)\Big) \tag{6}
      $$
   
   8. 消去合取词，把母式用子句集表示
      $$
      \bigg\{\Big(\neg P\big(x, f(x)\big), Q\big(x, g(x)\big)\Big), \Big(\neg P\big(x, f(x)\big), \neg R\big(x, g(x)\big)\Big)\bigg\} \tag{7}
      $$
   
   9. 子句变量标准化，即使每个子句中的变量符号不同
      $$
      \bigg\{\Big(\neg P\big(x, f(x)\big), Q\big(x, g(x)\big)\Big), \Big(\neg P\big(y, f(y)\big), \neg R\big(y, g(y)\big)\Big)\bigg\} \tag{8}
      $$
   
9. 利用归结反演方法来证明定理的具体步骤为：

   1. 否定目标公式 $G$，得到 $\neg G$；
   2. 将 $\neg G$ 并入到公式集 $F_1 \wedge F_2 \wedge \ldots \wedge F_n$ 中；
   3. 将公式集化子句集，得到子句集 $S$；
   4. 对 $S$ 进行归结，每次归结的结果并入到 $S$ 中。如此反复，直到得到空子句为止。此时，就证明了在前提 $F_1 \wedge F_2 \wedge \ldots \wedge F_n$ 为真时，结论 $G$ 为真。

10. 支持集策略：

    - 每次归结时，两个亲本子句中至少要有一个是目标公式否定的子句或其后裔。
    - 支持集 = 目标公式否定的子句集合 $\cup$ 这些子句通过归结生成的所有后裔子句

    特点：

    - 尽量避免在可满足的子句集中做归结，因为从中导不出空子句。而求证公式的前提通常是一致的，所以支持集策略要求归结时从目标公式否定的子句出发进行归结。支持集策略实际是一种目标制导的反向推理。
    - 支持集策略是完备的。
