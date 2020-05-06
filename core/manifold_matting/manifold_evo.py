from core.color_space_matting.color_space import ColorSpace
from core.data import MattingData
import numpy as np

from core.fitness import vanilla_fitness


def manifold_random(center, cluster, data: MattingData, color_space: ColorSpace, max_fes):
    max_fes = round(max_fes)
    pop_n = len(cluster)
    b_u = np.floor(np.random.sample(max_fes) * len(color_space.unique_color_b)).astype(int)
    d = np.random.sample(max_fes)
    best_f, best_b, best_alpha, best_fit, best_c = np.zeros(pop_n), np.zeros(pop_n), np.zeros(pop_n), np.zeros(pop_n), np.zeros(pop_n)
    for i, u in enumerate(cluster):
        s_u = data.s_u[u]
        rgb_u = data.rgb_u[u]
        md_fpu = data.min_dist_f[s_u[0], s_u[1]]
        md_bpu = data.min_dist_b[s_u[0], s_u[1]]
        b = color_space.u_color2n_id_b(b_u, u)
        b_u_rgb = color_space.unique_color_b[b_u].astype(int)
        f_u = color_space.ray_points_uid_f(b_u_rgb, u, d)
        f = color_space.u_color2n_id_f(f_u, u)
        alpha, fit, c, _, _ = vanilla_fitness(data.rgb_f[f], data.rgb_b[b], data.s_f[f], data.s_b[b], rgb_u, s_u, md_fpu, md_bpu)
        best = np.argmin(fit)
        best_f[i] = f[best]
        best_b[i] = b[best]
        best_alpha[i] = alpha[best]
        best_fit[i] = fit[best]
        best_c[i] = c[best]
    return best_f, best_b, best_alpha, best_c, best_fit


def manifold_evo(center, cluster, data: MattingData, color_space: ColorSpace, max_fes):
    pop_n = len(cluster)
    rgb_u = data.rgb_u[cluster]
    s_u = data.s_u[cluster]
    md_fpu = data.min_dist_f[s_u[:, 0], s_u[:, 1]]
    md_bpu = data.min_dist_b[s_u[:, 0], s_u[:, 1]]

    # initial random f/b id, and must be integer.
    b_u = np.random.randint(0, len(color_space.unique_color_b), pop_n)  # unique color id
    d = np.random.sample(pop_n)
    b = color_space.u_color2n_id_b4multi_u(b_u, cluster)  # color id
    b_u_rgb = color_space.unique_color_b[b_u].astype(int)

    f_u = color_space.ray_points_uid_f(b_u_rgb, cluster, d)
    f = color_space.u_color2n_id_f4multi_u(f_u, cluster)

    v_b = np.zeros([pop_n, data.color_channel])
    v_d = np.zeros(pop_n)

    alpha, fit, c, _, _ = \
        vanilla_fitness(data.rgb_f[f], data.rgb_b[b], data.s_f[f], data.s_b[b], rgb_u, s_u, md_fpu, md_bpu)

    best_f = f
    best_b = b
    best_alpha = alpha
    best_fit = fit
    best_c = c
    best_d = d

    fes = pop_n

    while fes < max_fes:
        shuffle_id = np.random.permutation(pop_n)
        random_pairs = np.array([shuffle_id[:round(pop_n / 2)], shuffle_id[len(shuffle_id) - round(pop_n / 2):len(shuffle_id)]])
        win = fit[random_pairs[0]] < fit[random_pairs[1]]
        winner = random_pairs[0] * win + random_pairs[1] * ~win
        loser = random_pairs[0] * ~win + random_pairs[1] * win

        v_random = np.random.rand(round(pop_n / 2), 1)
        d_random = np.random.rand(round(pop_n / 2), 1)
        c_random = np.random.rand(round(pop_n / 2), 1)

        # learning in rgb space.
        v_b[loser] = v_random * v_b[loser] + d_random * (b_u_rgb[winner] - b_u_rgb[loser]) + c_random * (data.rgb_b[best_b[loser]] - b_u_rgb[loser])
        v_d[loser] = v_random.squeeze() * v_d[loser] + d_random.squeeze() * (d[winner] - d[loser]) + c_random.squeeze() * (best_d[loser] - d[loser])

        b_u_rgb[loser] = v_b[loser] + b_u_rgb[loser]
        d[loser] = v_d[loser] + d[loser]

        # boundary control
        b_u_rgb[b_u_rgb > 255] = 255
        b_u_rgb[b_u_rgb < 0] = 0
        d[d < 0] = 0
        d[d > 1] = 1

        b_u[loser] = color_space.multi_rgb2unique_color_id(b_u_rgb[loser], color_space.color_space_b)
        b_u_rgb[loser] = color_space.unique_color_b[b_u[loser]].astype(int)

        f_u = color_space.ray_points_uid_f(b_u_rgb, cluster, d)
        f[loser] = color_space.u_color2n_id_f4multi_u(f_u[loser], cluster[loser])
        b[loser] = color_space.u_color2n_id_b4multi_u(b_u[loser], cluster[loser])

        alpha_loser, fit_loser, c_loser, _, _ \
            = vanilla_fitness(data.rgb_f[f[loser]], data.rgb_b[b[loser]], data.s_f[f[loser]],
                              data.s_b[b[loser]], rgb_u[loser], s_u[loser], md_fpu[loser], md_bpu[loser])

        fit[loser] = fit_loser
        alpha[loser] = alpha_loser
        c[loser] = c_loser

        better = fit < best_fit
        best_f[better] = f[better]
        best_b[better] = b[better]
        best_alpha[better] = alpha[better]
        best_fit[better] = fit[better]
        best_c[better] = c[better]
        best_d[better] = d[better]

        fes += round(pop_n / 2)

    # n x n
    # for i in range(pop_n):
    #     f_i = np.tile(f[i], pop_n)
    #     b_i = np.tile(b[i], pop_n)
    #     alpha_i, fit_i, c_i, _, _ = \
    #         vanilla_fitness(data.rgb_f[f_i], data.rgb_b[b_i], data.s_f[f_i], data.s_b[b_i], rgb_u, s_u, md_fpu, md_bpu)
    #
    #     better = fit_i < best_fit
    #     best_fit[better] = fit_i[better]
    #     best_f[better] = f_i[better]
    #     best_b[better] = b_i[better]
    #     best_alpha[better] = alpha_i[better]
    #     best_c[better] = c_i[better]

    return best_f, best_b, best_alpha, best_c, best_fit
