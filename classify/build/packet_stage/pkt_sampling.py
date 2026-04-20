import random

def sample_packets_by_target(cls_name, buckets, Target, ratios, seed):
    r_train, r_valid, r_test = ratios
    want_train = int(round(Target * r_train))
    want_valid = int(round(Target * r_valid))
    want_test  = Target - want_train - want_valid

    rnd = random.Random(seed)

    result = {"train": [], "valid": [], "test": []}

    # train
    tr = buckets["train"].copy()
    rnd.shuffle(tr)
    if len(tr) < want_train:
        return None, []
    result["train"] = tr[:want_train]

    # valid
    va = buckets["valid"].copy()
    rnd.shuffle(va)
    if len(va) < want_valid:
        return None, []
    result["valid"] = va[:want_valid]

    # test
    te = buckets["test"].copy()
    rnd.shuffle(te)
    if len(te) < want_test:
        return None, []
    result["test"] = te[:want_test]

    # only test overflow can be used for expansion
    test_overflow = te[want_test:]

    return result, test_overflow
