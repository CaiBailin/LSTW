import numpy as np

def H_2M(n, M):
    return (0.5 * (1 - np.cos(np.pi * n / M)))

def asym_window_init(window_size: int = True, hop_size: int = True, d: int = True):
    '''非对称窗初始化函数
    ---
    - window_size:  帧长或者窗长，一般指处理数据帧长度
    - hop_size:     帧移，一般指每次处理后移动长度
    - d:            分析窗前置0值长度,减轻混叠效应
    '''
    K = window_size
    M = hop_size
    ana_win = np.zeros(K)
    syn_win = np.zeros(K)

    ana_win[0:d]    = 0
    ana_win[d:K-M]  = np.sqrt(H_2M(np.arange(d, K-M) - d, K-M-d))
    ana_win[K-M:K]  = np.sqrt(H_2M(np.arange(K-M, K) - K + 2*M, M))
    syn_win[0:K-2*M]    = 0
    syn_win[K-2*M:K-M]  = H_2M(np.arange(K-2*M, K-M) - K + 2*M, M) / ana_win[K-2*M:K-M]
    syn_win[K-M:K]      = np.sqrt(H_2M(np.arange(K-M, K) - K + 2*M, M))

    return ana_win, syn_win


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # window example
    ana_win, syn_win = asym_window_init(window_size=512, hop_size=16, d=12)

    fig = plt.figure(num='asym window example')
    ax = fig.add_subplot(311)
    plt.plot(ana_win)
    plt.xlim(0, 512)
    plt.title('analysis window')
    ax = fig.add_subplot(312)
    plt.plot(syn_win)
    plt.xlim(0, 512)
    plt.title('synthesis window')
    ax = fig.add_subplot(313)
    plt.plot(ana_win* syn_win)
    plt.xlim(0, 512)
    plt.title('result=ana*syn')
    plt.tight_layout()
    plt.show()

    # use method
    sin_f = 50                      # Hz
    time_range = 0.3125             # s
    sig_sample_rate = 16000         # Hz
    point_num = int(time_range * sig_sample_rate)
    gif_frame_num = 500
    gif_update_interval = 100       # ms
    sin_signal = np.sin(2*np.pi*sin_f*np.linspace(0, time_range, point_num))
    ana_win, syn_win = asym_window_init(window_size=512, hop_size=16, d=12)
    rec_sin_signal = np.zeros_like(sin_signal)

    fig = plt.figure(num='sin({})'.format(sin_f))
    ax = fig.add_subplot(311)
    x = np.linspace(0, time_range, point_num)
    plt.plot(x, sin_signal)
    ax.set_xlim(0, time_range)
    ax.set_ylim(-1.2, 1.2)
    plt.title('input signal sin(2*pi*{}x) fs={}'.format(sin_f, sig_sample_rate))
    
    # animation of asymmetric windows
    ax = fig.add_subplot(312)
    line = ax.plot(x[0], sin_signal[0])[0]
    plt.title('OLA with asym win(LSTW), win size=512, hop size=16')
    ax.set_xlim(0, time_range)
    ax.set_ylim(-1.2, 1.2)
    def update(frame):
        if (frame >= (gif_frame_num-1)):
            exit("time over")
        global rec_sin_signal
        x_index = frame*16
        windowed_signal = sin_signal[x_index:(x_index+512)] * ana_win
        alg_out_signal = windowed_signal * syn_win
        # OLA 32=16*2 48=16*3
        if frame >= 1:
            rec_sin_signal[(x_index+512-32):(x_index+512)] = rec_sin_signal[(x_index+512-32):(x_index+512)] + alg_out_signal[-32:]
        else:
            rec_sin_signal[(x_index+512-32):(x_index+512)] = alg_out_signal[-32:]
        line.set_data(x, rec_sin_signal)
        return line
    #ani = FuncAnimation(fig, update, frames=range(gif_frame_num), interval=gif_update_interval)
    
    # animation of symmetric windows
    ax = fig.add_subplot(313)
    line_sym = ax.plot(x[0], sin_signal[0])[0]
    plt.title('OLA with sym win, win size=512, hop size=16')
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
    ana_win_sym, syn_win_sym = hann_window_sym(512, 16)
    rec_sin_signal_sym = np.zeros_like(sin_signal)
    def update_sym(frame):
        if (frame >= (gif_frame_num-1)):
            exit("time over")
        global rec_sin_signal_sym
        x_index_sym = frame*16
        windowed_signal_sym = sin_signal[x_index_sym:(x_index_sym+512)] * ana_win_sym
        alg_out_signal_sym = windowed_signal_sym * syn_win_sym
        if frame >= 1:
            rec_sin_signal_sym[x_index_sym:(x_index_sym+512)] += alg_out_signal_sym
        else:
            rec_sin_signal_sym[x_index_sym:(x_index_sym+512)] = alg_out_signal_sym
        line_sym.set_data(x, rec_sin_signal_sym)
    def update_api(frame):
        update(frame)
        update_sym(frame)
        return (line, line_sym)
    ani = FuncAnimation(fig, update_api, frames=range(gif_frame_num), interval=gif_update_interval)
    plt.tight_layout()
    ani.save(r'gif_out/sym_aym_window_compare.gif') # 打开保存
    plt.show()