"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np
from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)


def hamming74_encode(bits):
    """
    Hamming(7,4) 系统码编码。

    参数:
        bits: 一维 0/1 数组，长度必须是 4 的倍数。

    返回:
        encoded: 一维 0/1 编码比特数组，长度为输入的 7/4 倍。

    要求:
        使用课件中的生成矩阵 G，按 GF(2) 进行矩阵乘法。
    """
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    blocks = bits.reshape(-1, 4)
    codewords = (blocks @ HAMMING_G) % 2
    return codewords.reshape(-1)


def hamming74_syndrome(codewords):
    """
    计算 Hamming(7,4) 码字的伴随式。

    参数:
        codewords: 一维或二维 0/1 数组。若为一维，长度必须是 7 的倍数。

    返回:
        syndromes: 形状为 (N, 3) 的伴随式数组。
    """
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    syndromes = (codewords @ HAMMING_H.T) % 2
    return syndromes


def hamming74_decode(received):
    """
    Hamming(7,4) 单比特纠错译码。

    参数:
        received: 一维 0/1 接收序列，长度必须是 7 的倍数。

    返回:
        decoded_bits: 纠错后提取出的信息比特序列。

    提示:
        1. 计算每个码字的伴随式。
        2. 若伴随式非零，将其与 H 的各列比较，定位错误比特。
        3. 翻转对应错误位。
        4. 系统码的信息位为前 4 位。
    """
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    codewords = received.reshape(-1, 7).copy()
    syndromes = hamming74_syndrome(codewords)

    for i, syndrome in enumerate(syndromes):
        if np.any(syndrome != 0):
            for position in range(7):
                if np.array_equal(syndrome, HAMMING_H[:, position]):
                    codewords[i, position] ^= 1
                    break

    decoded_bits = codewords[:, :4]
    return decoded_bits.reshape(-1)


def _conv_next_state_and_output(state, input_bit):
    """
    (2,1,3) 卷积码状态转移和输出。
    生成多项式：
    g1 = 111
    g2 = 101

    state 用 2 bit 表示移位寄存器中的历史比特：
    state = b1 b2
    """
    previous_1 = (state >> 1) & 1
    previous_2 = state & 1

    output_1 = input_bit ^ previous_1 ^ previous_2
    output_2 = input_bit ^ previous_2

    next_state = (input_bit << 1) | previous_1

    return next_state, np.array([output_1, output_2], dtype=int)


def convolutional_encode(bits):
    """
    选做：实现 (2,1,3) 卷积码编码，生成多项式为 g1=111, g2=101。

    输入:
        bits: 一维 0/1 信息比特数组

    输出:
        encoded_bits: 一维 0/1 编码比特数组

    默认在末尾添加 2 个 0 作为尾比特，使状态回到全零。
    """
    bits = np.asarray(bits, dtype=int)

    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    state = 0
    outputs = []

    # 约束长度 K = 3，所以末尾补 2 个 0，让编码器回到全零状态
    terminated_bits = np.concatenate([bits, np.zeros(2, dtype=int)])

    for bit in terminated_bits:
        state, output = _conv_next_state_and_output(state, int(bit))
        outputs.extend(output)

    return np.asarray(outputs, dtype=int)


def viterbi_decode_hard(received_bits):
    """
    选做：实现 (2,1,3) 卷积码硬判决 Viterbi 译码。

    输入:
        received_bits: 一维 0/1 接收比特数组，长度必须为 2 的倍数

    输出:
        decoded_bits: 译码后的信息比特数组，不包含末尾 2 个尾比特
    """
    received_bits = np.asarray(received_bits, dtype=int)

    if received_bits.ndim != 1:
        raise ValueError('received_bits 必须是一维数组')

    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    if not np.all((received_bits == 0) | (received_bits == 1)):
        raise ValueError('received_bits 只能包含 0 或 1')

    received_pairs = received_bits.reshape(-1, 2)
    num_steps = len(received_pairs)
    num_states = 4

    # path_metrics[t, s] 表示第 t 步到达状态 s 的最小路径度量
    path_metrics = np.full((num_steps + 1, num_states), np.inf)
    predecessor_states = np.full((num_steps + 1, num_states), -1, dtype=int)
    predecessor_bits = np.full((num_steps + 1, num_states), -1, dtype=int)

    # 初始状态为 00
    path_metrics[0, 0] = 0.0

    for step, received_pair in enumerate(received_pairs, start=1):
        for state in range(num_states):
            if not np.isfinite(path_metrics[step - 1, state]):
                continue

            for input_bit in (0, 1):
                next_state, expected_pair = _conv_next_state_and_output(
                    state, input_bit
                )

                # 硬判决 Viterbi 使用汉明距离作为分支度量
                branch_metric = np.count_nonzero(received_pair != expected_pair)
                candidate_metric = path_metrics[step - 1, state] + branch_metric

                if candidate_metric < path_metrics[step, next_state]:
                    path_metrics[step, next_state] = candidate_metric
                    predecessor_states[step, next_state] = state
                    predecessor_bits[step, next_state] = input_bit

    # 因为编码时补了 2 个 0，理论上最终状态应该回到 00
    final_state = 0

    # 保险处理：如果最终 00 状态不可达，就选度量最小的状态
    if not np.isfinite(path_metrics[num_steps, final_state]):
        final_state = int(np.argmin(path_metrics[num_steps]))

    decoded_with_tail = []
    state = final_state

    for step in range(num_steps, 0, -1):
        bit = predecessor_bits[step, state]
        decoded_with_tail.append(bit)
        state = predecessor_states[step, state]

    decoded_with_tail.reverse()
    decoded_with_tail = np.asarray(decoded_with_tail, dtype=int)

    # 去掉编码时添加的 2 个尾比特
    if len(decoded_with_tail) >= 2:
        return decoded_with_tail[:-2]

    return decoded_with_tail


def run_coding_demo():
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for index, probability in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, probability, seed=100 + index)
            encoded_rx = binary_symmetric_channel(encoded, probability, seed=200 + index)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')


if __name__ == '__main__':
    run_coding_demo()
