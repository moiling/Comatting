## v0.1
- 2020.3.23

初始版本。<br/>
未知点加上九宫格内的点作为种群中的个体。认为邻近区域的解应该相近。<br/>
**决策变量**：[[F]\*9, [B]\*9]，**编码**:已知区域点的下标。<br/>
**初始化种群**：如果是已知区域点，用已知区域信息初始化（只有一个）；如果之前优化过该点，用之前最优解初始化；其余的随机初始化。<br/>
**进化策略**：类似CSO，随机选一半和另一半比较，适应值差的向适应值好的学习，保留每一个个体的当前速度。当前速度=之前的速度\*随机值+现在离好个体的距离\*随机值。<br/>
而且前景和背景分开学习，学习率前景=alpha\*距离，背景=(1-alpha)\*距离。因为认为越不透明，说明前景越好，越值得学习，背景越不值得学习。<br/>
```python
v_f[loser] = np.random.rand(round(pop / 2)) * v_f[loser] + alpha[winner] * np.random.rand(round(pop / 2)) * (f[winner] - f[loser])
v_b[loser] = np.random.rand(round(pop / 2)) * v_b[loser] + (1 - alpha[winner]) * np.random.rand(round(pop / 2)) * (f[winner] - f[loser])
```
**适应函数**：颜色，距离直接相加，距离除以当前点最小距离来平衡
```python
# Chromatic distortion
cost_c = np.sqrt(np.sum(np.square(rgb_u - (alpha[:, np.newaxis] * rgb_f + (1 - alpha[:, np.newaxis]) * rgb_b)), axis=1))
# Spatial cost
cost_sf = np.sqrt(np.sum(np.square(s_f - s_u), axis=1)) / (min_dist_fpu + 0.01)
cost_sb = np.sqrt(np.sum(np.square(s_b - s_u), axis=1)) / (min_dist_bpu + 0.01)
fit = (cost_c + cost_sf + cost_sb)
```
最后采用搜索过程中每个个体的最优解，并用每个个体的最优解套在其他个体身上，看看是否有更优的解（9x9）。


## v0.2
- 2020.3.24

BUG FIX:
1. B方向学习写错成了F分量了，且F\B分量随机不同，现统一。
```python
v_random = np.random.rand(round(pop / 2))
d_random = np.random.rand(round(pop / 2))
v_f[loser] = v_random * v_f[loser] + alpha[winner] * d_random * (f[winner] - f[loser])
v_b[loser] = v_random * v_b[loser] + (1 - alpha[winner]) * d_random * (b[winner] - b[loser])
```
2. 坏的向好的学习，学习的不是自己对应的好的……现修改如CSO，还提速了。
```python
winner = random_pairs[0] * win + random_pairs[1] * ~win
loser = random_pairs[0] * ~win + random_pairs[1] * win
```
其他修改：
1. 把shuffle函数换成了permutation函数，虽然没有变化，只有在很大的时候才能体现出速度优势，现在才9个数，速度降不下来。

试着把评价中F和B合并成X，结果速度还不如分开。


## v0.3
- 2020.3.24

写了个平滑……MATLAB转Python是真的难，还不知道哪出错了，改一天。

- 2020.3.25

轮廓标注

- 2020.3.26

random matting（无重复随机），效果还不错，可以作为基线标准。<br/>

- 2020.3.26-27

最普通版本的color space matting，类似CSO的学习机制，从FB空间学习转为在RGB空间学习，转换耗时严重<br/>
单点迭代，决策变量为[unique_F,unique_B]，转换离自己最近的F,B id进行Fitness计算，重新计算cost_c，不用FF'近似<br/>
学习时，学习优秀个体的RGB位置，然后映射回unique_F,unique_B，时间未做优化。<br/>
color space映射使用KNN，比grid data快10倍以上。<br/>
unique_F/B id转F/B id的时，临时找最近的F/B，耗时严重。<br/>
一开始代码写错了，导致最终结果alpha几乎都是0，但smooth结果却异常好，难道smooth本身也是抠图方法？传播？

- 2020.3.28

测试了用临近色分割效果，感觉作为初始解会很不错

- 2020.3.30

XY空间的学习，效果并不好。编写纯FB id的方法；编写XY空间+唯一颜色；编写随机+唯一颜色

- 2020.3.31

改代码，之前种群固定了100，那设置fes=10，也至少评价了100次，赖皮了<br/>
但该代码的时候发现spatial uc有个地方int没转型，之前是怎么跑成功的？（因为之前根本没进那个迭代）！<br/>
结果发现spatial uc比color space好，我之前的猜想好像站不住脚了<br/>

- 2020.4.1

把uc2id的方法提速了10倍

- 2020.4.2

把comatting的Window方法提速2倍

## v0.4
- 2020.4.3

射线方法，射线到颜色空间边缘的线上，没有去除重复颜色。不是alpha 0-255的方法。<br/>
b_ray_sample:b学习，f在bi射线上随机采样<br/>
b_ray:b学习，d也学习，d是指bi射线上到i的距离<br/>
结果证明b_ray_sample没b_ray好。

## v0.5
- 2020.4.5

多点评价，改正原来error计算错误（原来用uint8来算）


