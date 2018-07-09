import math
import numpy.random as rd
from numpy.random import *
import matplotlib.pyplot as plt

# 各種パラメータ
Kp = 0.44
Ki = 0.094
sigma = 0.1
# 学習率
h = 0.05
# 隠れ層ノード数
N = 55

# 入力層から隠れ層への重み
w1 = [list(rd.uniform(-1.0, 1.0, 4)) for i in range(N)]
# 隠れ層から出力層への重み，バイアスが含まれる
w2 = list(rd.uniform(-1.0, 1.0, (N+1)))
# 同定誤差
ek = 0.0

# u(k)，y(k)，v(k)，ε(k)
uk = [0.0, 0.0, 0.0]
yk = [0.0, 0.0, 0.0]
vk = [0.0, 0.0, 0.0]
epk = [0.0, 0.0]

# 結果を格納するリスト
costs = []
sys_outputs = []
nn_outputs = []


# 隠れ層の入出力関数f
def h_f(_x):
    return (1-math.exp(-_x))/(1+math.exp(-_x))


# 隠れ層のf'
def d_h_f(_x):
    return 2*math.exp(_x)/(math.exp(_x)+1)**2


# NNの評価関数
def calc_cost(): return 1/2*ek**2


# 目標値r(k)
def r(_k):
    if math.sin(2*math.pi*_k/50) > 0:
        return 1.0
    else:
        return 0.5


def update(_list):
    l_ = _list[0:len(_list)-1]
    l_.insert(0, 0.0)
    return l_


for k in range(0, 5000):
    # ノイズv(k):平均0,σ^2の正規分布に従う乱数で発生
    vk[0] = normal(0, sigma**2)
    # システム出力y(k)
    yk[0] = yk[1] - 0.35*yk[2] + 0.2*yk[1]**2 +\
                0.5*uk[1] + 0.1*uk[2] +\
                vk[0] - vk[1] + 0.35*vk[2]
    sys_outputs.append(yk[0])

    # NNによる出力ynnkを求める
    # NNの入力ベクトルx
    x = [yk[1], -0.35*yk[2], 0.5*uk[1], 0.1*uk[2]]
    v = [0.0 for i in range(N+1)]
    ynnk = 0.0
    for i, w21i in enumerate(w2):
        if i > 0:
            v[i] = h_f(sum([w1j*x[j] for j, w1j in enumerate(w1[i-1])]))
        else:
            v[i] = 1.0
        ynnk += w21i*v[i]
    nn_outputs.append(ynnk)

    # 同定誤差e(k)
    ek = yk[0] - ynnk
    costs.append(calc_cost())

    # 重みの更新
    for i, w21i in enumerate(w2):
        w21i += h*ek*1*v[i]
        if i > 0:
            for w1i in w1:
                for j, xj in enumerate(x):
                    w1i[j] += h*ek*1*w21i*d_h_f(sum([w1i[j]*xj for j, xj in enumerate(x)]))*xj

    # 目標値r(k)とシステム出力の誤差ε(k)
    epk[0] = r(k) - yk[0]
    d_epk = epk[0] - epk[1]

    # システムへの入力u(k)
    uk[0] = uk[1] + Kp*d_epk + Ki*epk[0]

    print('step: ' + str(k) + ' r(k): ' + str(r(k)) + ' y(k): ' + str(yk[0]))
    # print(calc_cost())

    # 各種値の更新（時刻kの値をシフト）
    yk = update(yk)
    uk = update(uk)
    epk = update(epk)

plt.figure()
plt.ylabel('cost function J')
plt.xlabel('sampling number k')
plt.bar([e for e in range(5000)], costs)
plt.ylim(ymax=0.1)
plt.savefig('costs.png')

plt.figure()
plt.ylabel('Output')
plt.xlabel('sampling number k')
plt.plot([e for e in range(4900, 5000)], sys_outputs[4900:len(sys_outputs)], label='y')
plt.plot([e for e in range(4900, 5000)], nn_outputs[4900:len(nn_outputs)], label='ynn')
plt.legend()
plt.savefig('outputs_4900-5000.png')

plt.figure()
plt.ylabel('Output')
plt.xlabel('sampling number k')
plt.plot([e for e in range(0, 100)], sys_outputs[0:100], label='y')
plt.plot([e for e in range(0, 100)], nn_outputs[0:100], label='ynn')
plt.legend()
plt.savefig('outputs_0-100.png')
