import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def deep_fir_window_init(window_size: int = True, hop_size: int = True):
    '''deep fir只要合成窗
    '''
    ana_win = np.hamming(window_size)
    if (hop_size % 2) != 0:
        print("hop size is error:{}".format(hop_size))
        return
    full_window = np.hanning(hop_size*2)
    syn_win_0 = full_window[:hop_size] # 前半部分
    syn_win_1 = full_window[hop_size:] # 后半部分

    err = np.ones_like(syn_win_0) - (syn_win_0+syn_win_1) # 补偿减少波形失真
    syn_win_1 += err

    return syn_win_0, syn_win_1

def gen_sin_signal():
    sin_signal = np.sin(2*np.pi*sin_f*np.linspace(0, time_range, point_num))

    return sin_signal

def demo_display(sin_signal):
    fig = plt.figure(num='sin({})'.format(sin_f))
    ax = fig.add_subplot(311)
    x = np.linspace(0, time_range, point_num)
    plt.plot(x, sin_signal)
    ax.set_xlim(0, time_range)
    ax.set_ylim(-1.2, 1.2)
    plt.title('input signal sin(2*pi*{}x) fs={}'.format(sin_f, sig_sample_rate))

    # deep fir 合成
    ax = fig.add_subplot(312)
    line = ax.plot(x[0], sin_signal[0])[0]
    plt.title('deep fir alg, win size={}, hop size={}'.format(win_size, hop_size))
    ax.set_xlim(0, time_range)
    ax.set_ylim(-1.2, 1.2)
    def deep_update(frame):
        # 如果帧数超过动画帧数退出
        if (frame >= (gif_frame_num-1)):
            exit("time over")
        global rec_sig # 重建信号
        x_index = frame*hop_size
        window_sig = test_sig[x_index:x_index+hop_size]
        windowed_sig_0 = window_sig * syn_left
        windowed_sig_1 = window_sig * syn_right
        windowed_sig = windowed_sig_0 + windowed_sig_1
        rec_sig[x_index:x_index+hop_size] = windowed_sig
        line.set_data(x, rec_sig) # 更新画面到画布上
        return line

    # 对称窗合成
    ax = fig.add_subplot(313)
    line_sym = ax.plot(x[0], sin_signal[0])[0]
    plt.title('OLA with sym win, win size={}, hop size={}'.format(win_size, hop_size))
    ax.set_xlim(0, time_range)
    ax.set_ylim(-1.2, 1.2)
    def hann_window_sym(fft_size, hop_size):
        '''symmetric window of hann
        '''
        block_num = fft_size // hop_size
        ana_win = np.zeros(fft_size)
        syn_win = np.zeros(fft_size)
        norm_win = np.zeros(fft_size)

        ana_win = np.hanning(fft_size)
        norm_win = ana_win * ana_win

        #for i in range(fft_size):
        #    ana_win[i] = (0.54 - 0.46*np.cos((2*i)*np.pi/(fft_size-1)))
    
        #for i in range(fft_size):
        #    norm_win[i] = ana_win[i] * ana_win[i]

        for i in range(hop_size):
            temp = 0
            for j in range(int(block_num)):
                temp += norm_win[i + j * hop_size]
            norm_win[i] = 1 / temp
    
        for i in range(hop_size):
            # 因为j=0的时候只是自身替换，没有意义，所以跳过这个处理过程
            for j in range(1, int(block_num)):
                norm_win[i + j * hop_size] = norm_win[i]

        for i in range(fft_size):
            syn_win[i] = norm_win[i] * ana_win[i]

        return ana_win, syn_win
    ana_win_sym, syn_win_sym = hann_window_sym(win_size, hop_size)
    # 对称窗更新函数
    def update_sym(frame):
        if (frame >= (gif_frame_num-1)):
            exit("time over")
        global rec_sin_signal_sym
        x_index_sym = frame*hop_size
        windowed_signal_sym = sin_signal[x_index_sym:(x_index_sym+win_size)] * ana_win_sym
        alg_out_signal_sym = windowed_signal_sym * syn_win_sym
        if frame >= 1:
            rec_sin_signal_sym[x_index_sym:(x_index_sym+win_size)] += alg_out_signal_sym
        else:
            rec_sin_signal_sym[x_index_sym:(x_index_sym+win_size)] = alg_out_signal_sym
        line_sym.set_data(x, rec_sin_signal_sym)
    def update_api(frame):
        deep_update(frame)
        update_sym(frame)
        return (line, line_sym)

    ani = FuncAnimation(fig, update_api, frames=range(gif_frame_num), interval=gif_update_interval)
    plt.tight_layout()
    ani.save(r'gif_out/deep_fir_compare.gif')
    plt.show()

if __name__ == '__main__':
    # 假设fir类型为allpass并且一直不变
    # 全局参数
    sin_f = 50                      # Hz
    time_range = 0.3125             # s
    sig_sample_rate = 16000         # Hz
    point_num = int(time_range * sig_sample_rate)
    gif_frame_num = 500             # gif总帧数
    gif_update_interval = 100       # ms gif动画频率
    win_size = 256                  # 分析窗长
    hop_size = 16                   # 合成窗长 # 1ms是绝大多数DSP更新间隔
    print("hop size is {}ms".format(hop_size/sig_sample_rate*1000))
    # 对称的合成窗
    syn_left, syn_right = deep_fir_window_init(window_size=win_size, hop_size=hop_size)
    # 测试信号
    test_sig = gen_sin_signal()
    rec_sig = np.zeros_like(test_sig)
    rec_sin_signal_sym = np.zeros_like(test_sig)
    # 显示
    demo_display(test_sig)