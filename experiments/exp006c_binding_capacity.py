from tac_sie.experiments import run_exp006c


if __name__ == "__main__":
    for n_pairs in [2, 3, 4, 5, 6]:
        print(n_pairs, run_exp006c(n_pairs=n_pairs))
