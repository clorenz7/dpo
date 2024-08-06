import numpy as np
import matplotlib.pyplot as plt


def plot_validation_curves(metrics, step_size=100, show_plot=False, save_plot=''):

    losses, win_rates, prob_deltas = [], [], []

    for data in metrics:
        if 'eval_loss' not in data:
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
    plt.plot(steps, losses)
    plt.ylabel('Loss')

    if len(win_rates) > 0:
        p_idx += 1
        plt.subplot(n_plots, 1, p_idx)
        plt.plot(steps, win_rates)
        plt.ylabel('Win Rate vs. Reject')

    if len(prob_deltas) > 0:
        p_idx += 1
        plt.subplot(n_plots, 1, p_idx)
        plt.plot(steps, prob_deltas)
        plt.ylabel('Log Prob Chosen - Reject')
    plt.xlabel('Step')

    if save_plot:
        plt.savefig(save_plot, pad_inches=0.02)

    if show_plot:
        plt.show()
