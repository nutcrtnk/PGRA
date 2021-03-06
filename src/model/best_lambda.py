best_lambda = {
    'dot': {
        'DBLP': {
            0.8: 0.003,
            0.6: 0.01,
            0.4: 0.03,
            0.2: 0.1,
        },
        'Yelp': {
            0.8: 0.001,
            0.6: 0.001,
            0.4: 0.01,
            0.2: 0.03,
        },
        'DM': {
            0.8: 0,
            0.6: 0,
            0.4: 0,
            0.2: 0.001,
        },
        'Aminer': {
            0.8: 0.01,
            0.6: 0.01,
            0.4: 0.03,
            0.2: 0.1,
        },
    },
    'l1': {
        'DBLP': {
            0.8: 0.001,
            0.6: 0.001,
            0.4: 0.01,
            0.2: 0.03,
        },
        'Yelp': {
            0.8: 0.001,
            0.6: 0.001,
            0.4: 0.001,
            0.2: 0.001,
        },
        'DM': {
            0.8: 0,
            0.6: 0.001,
            0.4: 0.001,
            0.2: 0.001,
        },
        'Aminer': {
            0.8: 0.01,
            0.6: 0.03,
            0.4: 0.03,
            0.2: 0.01,
        },
    },
    'l2': {
        'DBLP': {
            0.8: 0.003,
            0.6: 0.003,
            0.4: 0.001,
            0.2: 0.03,
        },
        'Yelp': {
            0.8: 0.003,
            0.6: 0.001,
            0.4: 0.001,
            0.2: 0.001,
        },
        'DM': {
            0.8: 0,
            0.6: 0.001,
            0.4: 0.001,
            0.2: 0.001,
        },
        'Aminer': {
            0.8: 0.01,
            0.6: 0.03,
            0.4: 0.03,
            0.2: 0.03,
        },
    }
}

def get_lambda(score, ds, n_test):
    return best_lambda[score][ds][n_test]
