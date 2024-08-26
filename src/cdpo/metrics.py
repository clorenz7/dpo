import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_validation_curves(metrics, step_size=100, show_plot=False,
                           save_plot='', log_loss=False, title=''):

    losses, win_rates, prob_deltas = [], [], []
    tr_losses = []

    for data in metrics:
        if 'eval_loss' not in data:
            if 'loss' in data:
                tr_losses.append(data['loss'])
            continue

        losses.append(data['eval_loss'])
        win_rate = data.get('eval_win_rate')
        if win_rate is not None:
            win_rates.append(win_rate)
        prob_delta = data.get('eval_avg_log_prob_delta')
        if win_rate is not None:
            prob_deltas.append(prob_delta)

    steps = np.arange(len(losses)) * step_size

    n_plots = 1
    if len(win_rates) > 0:
        n_plots += 1
    if len(prob_deltas) > 0:
        n_plots += 1

    if n_plots >= 1:
        plt.subplot(n_plots, 1, 1)
        p_idx = 1
    if log_loss:
        plt.semilogy(steps, losses, '.b-', steps[1:], tr_losses, '.g-')
    else:
        plt.plot(steps, losses, '.b-', steps[1:], tr_losses, '.g-')

    plt.ylabel('Loss')
    plt.grid(True, axis='y')
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)
    plt.legend(['Val Set', 'Train'])
    if title:
        plt.title(title)

    if len(win_rates) > 0:
        p_idx += 1
        plt.subplot(n_plots, 1, p_idx)
        plt.plot(steps, win_rates, '.b-')
        plt.ylabel('Win Rate vs. Reject')
        plt.grid(True, axis='y')
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)

    if len(prob_deltas) > 0:
        p_idx += 1
        plt.subplot(n_plots, 1, p_idx)
        plt.plot(steps, prob_deltas, '.b-')
        plt.ylabel(r'$\log \pi_W / \pi_L$')
        plt.grid(True, axis='y')
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)
    plt.xlabel('Step')

    if save_plot:
        plt.savefig(save_plot, pad_inches=0.02)
        plt.close()

    if show_plot:
        plt.show()
