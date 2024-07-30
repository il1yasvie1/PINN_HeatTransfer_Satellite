# 用PINN解决卫星热控问题

---

作者：刘三阳

日期：2024年7月30日

## 目录

1. 项目概述

2. 数学背景

3. 物理信息神经网络简介

4. $\varepsilon$ - 约束方法和同方差不确定性自平衡权重

---

## 1. 项目概述

预测卫星各个位置的温度对于维持卫星正常运作至关重要。卫星上的许多电子器件和设备都需要在特定的温度范围内才能正常工作。当卫星从地球的阴影面进入太阳辐射区时, 向阳面会迅速升温, 如果没有适当的散热措施, 卫星内部的仪器可能会受到影响。含热源的温度变化如下图所示：

<img title="" src="/README_source/边界热源.gif" alt="边界热源.gif" data-align="center">

此外, 太阳辐射在维持卫星温度方面也是一个重要的热源。当卫星位于地球的阴影面时, 它会向四周辐射热量, 导致温度急剧下降, 从而损耗电子元件的寿命。

<img title="" src="/README_source/无边界热源.gif" alt="无边界热源.gif" data-align="center">

本文致力于通过将卫星简化为单位正方体模型, 利用无数据监督的物理信息神经网络（PINN）预测卫星各个位置的温度分布。本文将使用COMSOL仿真数据作为验证集, 以进行对比测试和结果验证。PINN方法结合了物理定律与神经网络的计算能力, 能够在没有大量真实数据的情况下进行精确的温度预测, 从而为卫星的热控设计提供有力支持。

## 2. 数学背景

定义问题域为 $\Omega_\mathbf{x} = [0, 1]^3$ , 时间域为 $\Omega_t =[0, \infty)$。暴露在太阳辐射的表面为 $\partial\Omega_{x=1}=\{ (1,  y,  z):y,  z\in[0, 1]\}$ 。对于每个位置 $(x,  y,  z)\in \Omega_\mathbf{x}$ 和时间 $t\in \Omega_t$, 求满足以下偏微分方程的解 $T(x,  y,  z,  t)$：

$$
\begin{align*}
\rho C_p \partial_t T = \nabla_\mathbf{x} \cdot (k\nabla_\mathbf{x} T) ,  \quad & (\mathbf{x},  t) \in \Omega_{\mathbf{x}} \times \Omega_t \\
n \cdot (k\nabla_\mathbf{x} T) = \epsilon \sigma (T_{amb}^4 - T^4) ,  \quad & (\mathbf{x},  t)\in (\partial \Omega_\mathbf{x} \setminus \partial \Omega_{x=1}) \times \Omega_t\\
n \cdot (k\nabla_\mathbf{x} T) = Q_b,  \quad & (\mathbf{x},  t)\in \partial \Omega_{x=1} \times \Omega_t\\
T(\mathbf{x},  0)=T_0 , \quad & \mathbf{x}\in \Omega_\mathbf{x}
\end{align*}
$$

方程中的参数列于下表：

| 符号         | 含义                        | 值                   |
|:----------:|:------------------------- |:-------------------:|
| $\rho $    | density                   | 1                   |
| $C_p$      | heat capacity             | 1                   |
| $k$        | thermal conductivity      | 167                 |
| $\epsilon$ | emissivity                | 0.1                 |
| $\sigma$   | Stefan-Boltzmann Constant | $5.67\times10^{-8}$ |
| $T_{amb}$  | ambient temperature       | 3                   |
| $Q_b$      | boundary heat source      | 100                 |
| $T_0$      | initial temperature       | 300                 |

## 3. 物理信息神经网络简介

PINN使用了全连接神经网络`FCNN`, 网络中参数的初始化使用`xavier`方法, 代码如下：

```python
class FCNN(nn.Module):
    def __init__(self,  input_dimension,  output_dimension, 
                 n_hidden_layers,  neurons, 
                 regularization_param=0.,  regularization_exp=1., 
                 retrain_seed=40):
        super(FCNN,  self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.LeakyReLU()
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp

        self.input_layer = nn.Linear(self.input_dimension,  self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons,  self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons,  self.output_dimension)
        self.retrain_seed = retrain_seed
        self.init_xavier()

    def forward(self,  x):
        x = self.activation(self.input_layer(x))
        for k,  l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight,  gain=g)
                #torch.nn.init.xavier_normal_(m.weight,  gain=g)
                m.bias.data.fill_(0)
        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name,  param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param,  self.regularization_exp)
        return self.regularization_param * reg_loss
```

PINN计算域网格的生成通过 `generate_mesh` 函数实现。`generate_mesh` 函数生成的网格是一个六面体网格, 每个网格点在三维空间（x,  y,  z）和时间维度（t）上均匀分布。代码如下：

```python
def generate_mesh(N,  Nt,  t_max=1,  no_interior=True):
    linspace = torch.linspace(0,  1,  N)
    x,  y,  z,  t = torch.meshgrid(linspace,  linspace,  linspace,  torch.linspace(0,  t_max,  Nt),  indexing='ij')
    mesh = torch.stack([x,  y,  z,  t],  dim=-1  ).reshape(-1,  4)
    if no_interior:
        mask = (mesh[:,  0] == 0) | (mesh[:,  0] == 1) | (mesh[:,  1]==0) | (mesh[:,  1]==1) | (mesh[:,  2]==0) | (mesh[:,  2]==1)
        mesh = mesh[mask]
    return mesh
```

## 4. $\varepsilon$ -约束方法和同方差不确定性自平衡权重

在该问题中, 根据模型假设, 所求温度函数 $T(x, y, z, t)$ 在 $t=0$ 处不连续。神经网络的谱偏差性质表明, 神经网络对频谱之中的低频成分响应更快, 而高频部分响应更慢。通过可视化, 发现该问题的解的确有高频的间断现象发生, 如下图：

<img title="" src="/README_source/init.png" alt="对比.png" width="701" data-align="center">

可以看到, 温度在第一个时间步迅速从初始温度均匀分布的273K变化至263~369K。而使用全连接神经网路的PINN在不经过任何修改的情况下难以对间断函数进行响应。

### 4.1 $\varepsilon$ -约束法

为解决以上问题, 类比解决多目标优化问题里的 $\varepsilon$ -约束法, 本文设计出一种关于初始条件的松弛约束, 并将其加入损失函数。

首先, 多目标优化是一种优化方法, 旨在同时优化多个目标函数。在实际应用中, 往往需要在相互冲突的目标之间找到一个平衡点。例如, 在产品设计中, 我们可能需要同时考虑成本和性能, 这两个目标通常是相互冲突的。在PINN的损失随训练下降过程中, 各项损失函数例如PDE损失、初始条件损失、边界条件损失等等可视为待优化的多个目标函数。在进行下降的过程中, 各个目标时而会展现出矛盾或相互冲突的趋势。因此, 我们可以将PINN的各项损失函数优化问题类比为多目标优化。

**Pareto 最优**（Pareto optimality）是多目标优化中的一个重要概念。在多个目标函数中, 若某一解不存在其他解使得所有目标都更优（至少一个目标更优且其他目标不变）, 则该解为Pareto 最优解。换句话说, 任何进一步优化某个目标的尝试都会使其他一个或多个目标变得更差。求多目标问题的Pareto最优有很多方法, 而 $\varepsilon$ -约束法相较于其他方法在PINN中实现简单, 并且通过设置合适的 $\varepsilon$ 阈值能够确保最优解为Pareto最优。

假设我们包括两个目标的多目标优化问题 $\min (f_1(x),  f_2(x))$ , 其中 $f_{1}(x),  f_{2}(x) > 0$ , 利用 $\varepsilon$ -约束法, 可将问题转化为 $\min \{f_1(x) :f_2(x)<\varepsilon\}$ 。在PINN中, 我们类比以上过程, 设计新的损失函数:

$$
\begin{align*}
L_{pde} &= \frac{1}{|\Omega|} \sum_{(x, y, z, t)\in \Omega}  | \rho C_p \partial_t T_{\mathcal{NN}} - k\Delta_{x,y,z} T_{\mathcal{NN}} |^2 \\
L_{\Gamma_i} &= \frac{1}{|\Gamma_i|} \sum_{(x, y, z, t) \in \Gamma_i}  |  k\nabla_{x,y,z} T_{\mathcal{NN}} \cdot n - \epsilon \sigma (T_{amb}^4 - T_{\mathcal{NN}}^4) |^2 \\
L_{IC} &= \frac{1}{| \Omega_{x,y,z}|} \sum_{(x, y, z) \in \Omega_{x,y,z}} \max(|  T_{\mathcal{NN}} - T_0|^2 - \varepsilon， 0)
\end{align*}
$$

可以看到,  在下降的过程中,  $\varepsilon$-约束法对初始条件进行了松弛化,  而同时也保证解的Pareto最优性。然而,  松弛约束的同时,  我们需要一个自适应权重方法,  来提高初始条件在训练初期的重要性,  并逐渐将其降低,  以让神经网络响应间断点。

### 4.2 同方差不确定性自平衡权重法（Homoscedastic Task Uncertainty for Loss Weighting）

在总损失函数设计方面, 使用了同方差不确定性自平衡权重法（Homoscedastic Task Uncertainty for Loss Weighting）。在多任务学习中, 同方差任务不确定性（Homoscedastic Task Uncertainty）是一种用于对不同任务的损失进行加权的方法。它通过引入任务固有的不确定性来自动调整各任务的损失权重, 从而平衡它们在总损失中的贡献。下面我们引入任务固有的不确定性参数 $\sigma_i$ , 定义加权总损失函数：

$$
\mathcal{L}_ {total} = \sum_i \frac{1}{2\sigma_i} \mathcal{L}_i + \log(\sigma_i)
$$

在优化的过程中, 应对初始条件设置一个较大的初始权重, 即较小的$\sigma_{IC}$, 使得网络在一开始可以迅速响应初始条件, 即高频间断部分；并后续过程中通过 $\varepsilon$ -约束逐渐学习低频部分。

### 4.3 代码实现

```python
class PINN:
    def __init__(self,  lambdas, 
                 n_hidden_layers=3,  neurons=8, 
                 T0    = 273., 
                 rho   = 1., 
                 C_rho = 1., 
                 k     = 1, 
                 Qb    = 100., 
                 epsilon   = 0.1, 
                 sigma = 5.67e-8, 
                 T_amb = 3.
                ):
        """
        初始化 PINN 类, 设置损失函数权重和物理参数。

        参数：
        - lambdas: 损失函数的权重
        - n_hidden_layers: 隐藏层数量
        - neurons: 每个隐藏层的神经元数量
        - T0: 初始温度
        - rho: 密度
        - C_rho: 比热容
        - k: 热导率
        - Qb: 边界热流
        - epsilon: 辐射率
        - sigma: 斯蒂芬-玻尔兹曼常数
        - T_amb: 环境温度
        """

        # 设置损失函数权重, 并将其添加到待训练参数中
        self.lambdas = lambdas.clone().detach().requires_grad_(True)
        self.params_to_train = [self.lambdas]

        # 初始化神经网络模型
        self.u = FCNN(input_dimension=4,  output_dimension=1, 
                         n_hidden_layers=n_hidden_layers,  neurons=neurons, 
                         regularization_param=1.,  regularization_exp=2., 
                         retrain_seed=0)

        # 设置物理参数
        self.T0 = T0
        self.rho = rho
        self.C_rho = C_rho
        self.k = k
        self.Qb = Qb
        self.epsilon = epsilon
        self.sigma = sigma
        self.T_amb = T_amb

    def set_lambdas(self,  lambdas):
        """
        更新损失函数权重。

        参数：
        - lambdas: 新的损失函数权重
        """
        self.lambdas = lambdas

    def predict(self,  Xt):
        """
        使用神经网络模型对输入数据进行预测。

        参数：
        - Xt: 输入数据

        返回：
        - 神经网络的预测结果
        """
        return self.u(Xt)

    def compute_losses(self,  Xt_mesh):
        """
        计算损失函数, 包括 PDE 残差、初始条件残差和边界条件残差。

        参数：
        - Xt_mesh: 包含空间和时间的网格数据

        返回：
        - total_loss: 总损失值
        - losses: 各项损失值的集合
        """
        Xt_mesh.requires_grad = True

        # 预测值和梯度
        uXt = self.predict(Xt_mesh)
        grad = autograd(uXt,  Xt_mesh)[0]
        Du = grad[:,  :3]
        Dt = grad[:,  -1].reshape(-1,  1)
        Dx,  Dy,  Dz = Du.T
        Dx = Dx.reshape(-1,  1)
        Dy = Dy.reshape(-1,  1)
        Dz = Dz.reshape(-1,  1)
        Dxx = autograd(Dx,  Xt_mesh)[0].T[0].reshape(-1, 1)
        Dyy = autograd(Dy,  Xt_mesh)[0].T[1].reshape(-1, 1)
        Dzz = autograd(Dz,  Xt_mesh)[0].T[2].reshape(-1, 1)
        Xt_mesh.requires_grad = False

        # PDE 残差
        loss_pde = torch.mean((self.rho * self.C_rho * Dt - self.k * (Dxx + Dyy + Dzz))**2)

        # 初始条件残差
        mask = (Xt_mesh[:,  -1] == 0)
        loss_ic = max(torch.mean((uXt[mask] - self.T0) ** 2) - 1e-1,  torch.tensor(0.0))

        # 边界条件残差
        # x=0 边界条件
        mask = (Xt_mesh[:,  0] == 0) 
        loss_bc = torch.mean((-self.k * Dx[mask] - self.epsilon * self.sigma * (self.T_amb**4 - uXt[mask]**4))**2)

        # x=1 边界条件
        mask = (Xt_mesh[:,  0] == 1) 
        loss_bc_s = torch.mean((self.k * Dx[mask] - self.Qb)**2)

        # y=0 边界条件
        mask = (Xt_mesh[:,  1] == 0) 
        loss_bc += torch.mean((-self.k * Dy[mask] - self.epsilon * self.sigma * (self.T_amb**4 - uXt[mask]**4))**2)

        # y=1 边界条件
        mask = (Xt_mesh[:,  1] == 1) 
        loss_bc += torch.mean((self.k * Dy[mask] - self.epsilon * self.sigma * (self.T_amb**4 - uXt[mask]**4))**2)

        # z=0 边界条件
        mask = (Xt_mesh[:,  2] == 0) 
        loss_bc += torch.mean((-self.k * Dz[mask] - self.epsilon * self.sigma * (self.T_amb**4 - uXt[mask]**4))**2)

        # z=1 边界条件
        mask = (Xt_mesh[:,  2] == 1) 
        loss_bc += torch.mean((self.k * Dz[mask] - self.epsilon * self.sigma * (self.T_amb**4 - uXt[mask]**4))**2)

        # 汇总所有损失
        losses = torch.hstack([loss_pde,  loss_ic,  loss_bc,  loss_bc_s])
        total_loss = (1/(2*self.lambdas**2)) @ losses + torch.log(self.lambdas).sum().reshape(-1, )
        return total_loss,  losses

    def train(self,  Xt_mesh, 
             epochs=1000,  lr=1e-3, 
             verbose=True):
        """
        训练 PINN 模型。

        参数：
        - Xt_mesh: 包含空间和时间的网格数据
        - epochs: 训练周期数
        - lr: 学习率
        - verbose: 是否显示训练过程中的详细信息

        返回：
        - history: 总损失的历史记录
        - history_pde: PDE 残差的历史记录
        - history_ic: 初始条件残差的历史记录
        - history_bc: 边界条件残差的历史记录
        - history_bc_s: 边界条件残差（x=1）的历史记录
        - errs: 相对误差的历史记录
        """
        # 优化器
        opt = optim.Adam([{'params': self.u.parameters()},  {'params': self.params_to_train}],  lr=lr)

        errs = []
        history = []
        history_pde = []
        history_ic = []
        history_bc = []
        history_bc_s = []

        for epoch in range(epochs):
            opt.zero_grad()
            total_loss,  losses = self.compute_losses(Xt_mesh)
            loss_pde,  loss_ic,  loss_bc,  loss_bc_s = losses
            total_loss.backward()
            opt.step()

            with torch.no_grad():
                if verbose:
                    percentage = round(100 * epoch / epochs,  1)
                    losses_str = ',  '.join([f"{loss:.4f}" for loss in losses.cpu().numpy()])
                    lambdas_str = ',  '.join([f"{l:.4f}" for l in self.lambdas.cpu().numpy()])
                    print(f"\r[{percentage}%] losses: [{losses_str}] | lambdas: [{lambdas_str}] | total loss: [{total_loss.item()}]",  end='')

            errs.append(relative_error(self.u(Xt),  U))
            history.append(total_loss.item())
            history_pde.append(loss_pde.item())
            history_ic.append(loss_ic.item())
            history_bc.append(loss_bc.item())
            history_bc_s.append(loss_bc_s.item())

        return history,  history_pde,  history_ic,  history_bc,  history_bc_s,  errs
```

### 4.4 结果对比

下面, 我们将对比改进前后的结果。在相同的数据集上, 用同样的训练方式, 普通PINN的测试误差为5.32%, 而改进的PINN的测试误差为1.99%。

首先, 我们先对比两种方法的各项损失随训练下降曲线。

<img title="" src="/README_source/train_loss.png" alt="对比1.png" width="732" data-align="center">

可以看到, 普通PINN的总损失曲线整体呈先下降, 后趋于平稳的趋势；而改进后的PINN很快便趋近于0。

接下来, 在训练过程中, 我们还记录了在测试集上的误差和百分比训练损失的对比图, 由这张图可以验证损失函数的准确性。如下图：

<img title="" src="/README_source/loss_error.png" alt="对比2.png" data-align="center" width="670">

可以看到, 两种PINN的损失函数的下降趋势均符合测试误差下降趋势。在普通PINN中, 可以看到, 在大约第50epoch中, 测试误差下降有震荡, 而损失函数也体现相应震荡；而在改进中的PINN中, 训练损失下降曲线更为平滑, 受震荡影响较小。

下面我们通过可视化温度随时间的变化对比两种方法。首先, 我们先观察普通PINN的结果。如下图：

![anime_pinn.gif](\README_source\pinn.gif)

可以看出, 在 $t=0$ 时, 左图真实值为273K均匀分布；而右图中, 温度并不处于均匀分布状态, 而更加接近下一个时间步的真实值。温度在初始时刻的剧烈变化未能被准确反映, 进而影响了后续时间步的温度预测。这种不准确的初始条件处理直接导致了后续温度场的耗散速度过快, 使得模型在预测过程中逐步偏离真实情况。

接下来, 观察改进后的PINN的结果：

<img title="" src="/README_source/pinn_mod.gif" alt="anime_pinn_modified.gif" data-align="center">

可以看出, 在初始条件方面, 相比于普通PINN, 改进后的PINN增强了网络对高频间断的学习能力, 与真实温度的均匀分布更加接近。具体来说, 通过引入 $\varepsilon$ -约束和同方差任务不确定性自平衡权重法, 改进后的PINN在处理初始条件的间断性时表现出了更强的适应性。

这种改进的直接效果是显而易见的。改进后的PINN在初始时刻能够迅速响应高频变化, 使得温度分布更加接近真实情况, 避免了普通PINN中出现的温度耗散过快的问题。通过增强对初始条件高频成分的学习能力, 改进后的PINN不仅提高了初始条件的拟合精度, 也为后续时间步的温度预测奠定了更好的基础。结果显示, 改进后的PINN在整个预测过程中表现出更高的精度和稳定性, 有效地解决了普通PINN在处理初始条件时的不足之处。
