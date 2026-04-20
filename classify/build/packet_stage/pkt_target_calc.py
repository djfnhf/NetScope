def compute_target(P_train, P_valid, P_test, ratios, pkt_min, pkt_cap):
    r_train, r_valid, r_test = ratios

    if P_train == 0 or P_valid == 0 or P_test == 0:
        return 0

    T1 = P_train / r_train
    T2 = P_valid / r_valid
    T3 = P_test  / r_test

    Target = int(min(T1, T2, T3))  

    if Target < pkt_min:
        return 0

    if Target > pkt_cap:
        Target = pkt_cap

    return Target
