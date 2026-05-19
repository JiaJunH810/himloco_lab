import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class HIMEstimator(nn.Module):
    """HIM (History-Informed) 隐式运动状态估计器。

    核心要解决的问题：
    - Actor 不能访问特权信息（真实速度、地形高度等），这确保了 sim-to-real 可迁移
    - 但 Actor 需要知道自己的速度、运动状态才能做好控制
    - Estimator 从观测历史中"推断"速度和隐变量，作为 Actor 的额外输入

    三个子网络及其角色：
    1. Encoder:  观测历史(6步×45=270维) → 预测速度(3) + 隐变量 z_s(16)
       从 6 步的时序变化推断当前运动状态
    2. Target:   当前实际观测(45维) → 目标隐变量 z_t(16)
       直接看当前一帧（含真实速度），为 Encoder 提供对比学习的目标
    3. Prototype: 32个可学习的16维向量，作为"运动状态类别"的聚类原型
       每个 prototype 代表一类运动状态（如：平地慢走、上坡快跑、台阶跌倒等）

    z_s 和 z_t 编码的是什么：
    - 不是单纯的"地形环境"，而是更广义的"运动状态摘要"
    - 包含：速度状态（静止/慢走/快跑）、姿态（水平/前倾/侧倾）、
      关节构型（蹲姿/站姿/摆动相）、地形特征（从关节运动模式和 gravity 向量推断）、
      步态相位（哪只脚支撑、哪只脚摆动）等
    - Encoder 和 Target 看的是不同的输入（历史 vs 当前+特权信息），
      但编码的是同一时刻的运动状态

    两个训练目标：
    - estimation_loss: MSE(预测速度, 真实速度)
      监督信号来自 Critic 特权观测中的 base_lin_vel
      Actor 部署时没有速度传感器，但通过这个损失，Encoder 学会了从历史推断速度
    - swap_loss: SwAV 风格互换预测损失
      让 Encoder 和 Target 在 Prototype 空间中产生一致的聚类分配
      Encoder 看不到特权信息，但被迫和 Target 产生相同的聚类结果
      → Encoder 被迫从历史中隐式推断出速度、地形等信息

    为什么必须用 Sinkhorn 防止坍塌：
    - 如果不用 Sinkhorn，所有样本都会被分给同一个 prototype
      → Encoder 对任何输入都输出差不多的 z_s，Target 也输出差不多的 z_t
      → swap loss 虽然很低，但 16 维隐变量完全没有区分度（假收敛）
    - Sinkhorn 强制均匀分配：不同运动状态必须映射到不同的 prototype
      → Encoder 被迫对不同的输入产生不同的 z_s
      → z_s 必须有真正的区分度，隐式编码了速度、地形等信息
      → Actor 才能拿到有意义的额外输入

    核心逻辑链条：
    Sinkhorn 强制均匀分配
      → Encoder 必须对不同输入产生不同输出
      → z_s 必须有区分度（有信息量）
      → Encoder 被迫从历史中隐式推断速度、地形等
      → Actor 拿到有效的隐变量
      → sim-to-real 可迁移

    初始化参数：
        temporal_steps:  观测历史步数 (6)
        num_one_step_obs: 单步观测维度 (45)
        enc_hidden_dims:  Encoder 隐藏层维度 [128, 64, 16]
        tar_hidden_dims:  Target 隐藏层维度 [128, 64]
        num_prototype:    原型向量数量 (32)，即运动状态类别数
        temperature:      softmax 温度 (3.0)，值越大分布越平滑
    """
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps        # 6 步历史
        self.num_one_step_obs = num_one_step_obs    # 45 维 (policy obs)
        self.num_latent = enc_hidden_dims[-1]       # 16 维隐变量
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature              # softmax 温度 τ=3.0

        # ==================== Encoder：观测历史 → 预测速度 + 隐变量 ====================
        # 输入: 6步 × 45维 = 270维 (Actor 观测历史)
        # 输出: 19维 = pred_vel(3) + latent(16)
        enc_input_dim = self.temporal_steps * self.num_one_step_obs  # 270
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        # 最后一层：输出 16(隐变量) + 3(速度) = 19 维
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)   # 270 → 128 → 64 → 19

        # ==================== Target：当前实际观测 → 目标隐变量 ====================
        # 输入: 45维 (从 Critic 特权观测中切出的"实际发生"的单步观测)
        #       = base_ang_vel(3) + projected_gravity(3) + joint_pos_rel(12)
        #       + joint_vel_rel(12) + last_action(12) + base_lin_vel(3)
        #       注意：velocity_commands 被 base_lin_vel 替换了
        # 输出: 16维隐变量 z_t
        tar_input_dim = self.num_one_step_obs  # 45
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)     # 45 → 128 → 64 → 16

        # ==================== Prototype：32 个"运动状态类别"原型 ====================
        # 每个 prototype 是一个 16 维向量，共 32 个，等价于 32 种运动状态模板
        # 希望不同运动状态（快跑/慢走/爬坡/台阶...）激活不同的 prototype
        # 使用 nn.Embedding 实现，等价于 torch.nn.Parameter(randn(32, 16))
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])  # [32, 16]

        # ==================== 独立优化器 ====================
        # Estimator 有自己的 Adam 优化器，与 PPO 的优化器独立
        # 学习率动态跟随 PPO 学习率 (在 update() 中同步)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        """仅获取隐变量和预测速度 (训练/推理时的只读接口)。"""
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        """推理前向传播：观测历史 → 预测速度 + 隐变量 (皆 detach，不参与梯度)。

        输入:  obs_history [batch, 270]  Actor 观测历史 (6步 × 45维)
        输出:  vel        [batch, 3]     预测的机身速度
              z          [batch, 16]     L2归一化后的隐变量
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]  # 拆分：前3维=速度，后16维=隐变量
        z = F.normalize(z, dim=-1, p=2)          # L2归一化到单位球面
        return vel.detach(), z.detach()           # detach: 推理时不建计算图

    def encode(self, obs_history):
        """编码观测历史 (训练时使用，保留梯度)。

        与 forward 的区别：不 detach，保留梯度，用于 update() 中计算梯度。
        """
        parts = self.encoder(obs_history.detach())  # obs_history 本身仍 detach
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        """Estimator 的一次训练更新。

        Args:
            obs_history:     [batch, 270]  t 时刻 Actor 观测历史 (6步 × 45维)
            next_critic_obs: [batch, 1266] t+1 时刻 Critic 观测历史 (6步 × 211维)
            lr:              可选，动态同步 PPO 学习率

        Returns:
            estimation_loss.item(): 速度预测 MSE
            swap_loss.item():       SwAV 互换预测损失

        === 数据提取说明 ===
        next_critic_obs 最前面一步的结构 (step_t+1, 共 211 维):
          [0:3]    velocity_commands
          [3:6]    base_ang_vel
          [6:9]    projected_gravity
          [9:21]   joint_pos_rel
          [21:33]  joint_vel_rel
          [33:45]  last_action
          [45:48]  base_lin_vel     ← 真实速度标签 vel
          [48:51]  base_external_force
          [51:211] height_scanner

        Target 网络输入 next_obs = [3:48]:
          base_ang_vel(3) + projected_gravity(3) + joint_pos_rel(12)
          + joint_vel_rel(12) + last_action(12) + base_lin_vel(3) = 45维
          (velocity_commands 被替换为 base_lin_vel —— Target 看到的是"实际发生了什么")
        """
        # ── 同步 PPO 学习率 ──
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        # ── 数据切割 ──
        # 真实速度标签: 取最前面一步 Critic 观测的第 45-47 维 = base_lin_vel(3)
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()  # [batch, 3]

        # Target 网络输入: 取最前面一步 Critic 观测的第 3-47 维 = 45 维
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]                 # [batch, 45]

        # ── 前向传播 ──
        z_s = self.encoder(obs_history)   # Encoder: 历史 → 19维 (pred_vel[3] + z_s[16])
        z_t = self.target(next_obs)       # Target: 当前实际观测 → 16维 (z_t[16])
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]  # 拆分 Encoder 输出

        # ── L2 归一化到单位球面 ──
        # 归一化后内积 = 余弦相似度，值域 [-1, +1]
        # z_s: Encoder 从 6 步历史推断的当前运动状态
        # z_t: Target 从当前一帧（含真实速度）看到的实际运动状态
        # 两者来自不同的输入视角，但编码的是同一时刻的同一运动状态
        z_s = F.normalize(z_s, dim=-1, p=2)  # [batch, 16]
        z_t = F.normalize(z_t, dim=-1, p=2)  # [batch, 16]

        # Prototype 也归一化到单位球面 (直接修改参数值，不参与梯度)
        with torch.no_grad():
            w = self.proto.weight.data.clone()    # [32, 16]
            w = F.normalize(w, dim=-1, p=2)       # 每个 prototype 模长 = 1
            self.proto.weight.copy_(w)

        # ── 相似度矩阵 ──
        # score[i, j] = 样本 i 与 prototype j 的余弦相似度
        score_s = z_s @ self.proto.weight.T  # [batch, 16] @ [16, 32] = [batch, 32]
        score_t = z_t @ self.proto.weight.T  # [batch, 16] @ [16, 32] = [batch, 32]

        # ── Sinkhorn-Knopp: 将相似度转为均匀分配的软标签 ──
        # 不参与梯度（no_grad），仅作为 swap loss 中的"目标分布"
        #
        # 为什么必须用 Sinkhorn 防止坍塌：
        # 如果没有 Sinkhorn，所有样本都会坍缩到 1-2 个 prototype
        #   → Encoder 对任何输入都输出差不多的 z_s，Target 也输出差不多的 z_t
        #   → swap loss 很低，但隐变量没有任何区分度（假收敛！）
        #   → Actor 拿到的 latent 是无用信息
        # Sinkhorn 强制"不同运动状态映射到不同 prototype"
        #   → Encoder 被迫对快跑/慢走/爬坡/台阶等不同状态产生不同的 z_s
        #   → z_s 必须有真正的区分度，隐式编码速度、地形等信息
        #   → Actor 拿到有意义的隐变量
        with torch.no_grad():
            q_s = sinkhorn(score_s)  # [batch, 32] Encoder 侧的软标签
            q_t = sinkhorn(score_t)  # [batch, 32] Target 侧的软标签

        # ── Softmax 概率 (带温度) ──
        # 温度 τ=3.0 让分布偏软，保证 swap loss 有足够的梯度
        # 如果 τ 太小（硬标签），与 q 对齐时梯度太弱，学不动
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)  # [batch, 32]
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)  # [batch, 32]

        # ── Swap Loss: 互换预测 ──
        # Encoder 的标签 (q_s) 监督 Target 的预测 (log_p_t)
        # Target 的标签 (q_t) 监督 Encoder 的预测 (log_p_s)
        # 由于 Encoder 和 Target 看的是不同输入（历史 vs 当前+特权），
        # 两者必须通过各自的方式达到相同的 prototype 分配
        # → Encoder 被迫从历史中隐式推断出 Target 靠特权信息才能得到的结果
        # → 隐变量学会了在没有速度传感器的情况下编码速度信息
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

        # ── 速度预测损失 ──
        estimation_loss = F.mse_loss(pred_vel, vel)

        # ── 总损失 ──
        losses = estimation_loss + swap_loss

        # ── 反向传播 ──
        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    """Sinkhorn-Knopp 算法：将相似度矩阵变为双重随机矩阵。

    输入:  out  [batch, K]  相似度矩阵 (K=32 个 prototype)
    eps:    0.05    温度参数，越小分布越硬 (趋向 one-hot)
    iters:  3       迭代次数

    输出:  [batch, K]  双重随机矩阵
          每行 (每个样本): 和 = 1/K  (均匀分配到 K 个 prototype)
          每列 (每个 prototype): 和 = 1/B (每个 prototype 被 B 个样本均匀使用)

    算法流程:
    1. exp(out/eps) 放大微小差异，转置为 [K, batch]
    2. 归一化总和 = 1
    3. 交替归一化行和列 (迭代 iters 次)
       - 行归一化: 每行 ÷行和 ÷K → 每个 prototype 的"配额"均等
       - 列归一化: 每列 ÷列和 ÷B → 每个样本的"分配量"均等
    4. 转置回 [batch, K] 并缩放

    核心作用: 强制 32 个 prototype 全部被均匀使用。
    如果不用 Sinkhorn，所有运动状态会坍缩到 1-2 个 prototype，
    Encoder 对任何输入都输出相同的 z_s，隐变量完全丧失区分度（假收敛）。
    Sinkhorn 强制不同运动状态激活不同 prototype → z_s 必须有信息量 → Actor 获益。
    """
    Q = torch.exp(out / eps).T          # [K, batch]  指数放大差异
    K, B = Q.shape[0], Q.shape[1]       # K=32 prototype, B=batch

    Q /= Q.sum()                        # 总和归一化到 1

    for it in range(iters):
        # 行归一化: 总权重 per prototype = 1/K (每个 prototype 配额均等)
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # 列归一化: 总权重 per sample = 1/B (每个样本分配量均等)
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    return (Q * B).T                     # [batch, K]  缩放回 B 尺度后转置


def get_activation(act_name):
    """根据名称返回对应的激活函数。"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
