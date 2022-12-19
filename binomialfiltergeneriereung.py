import numpy as np



if __name__ == '__main__':
    sqares = 16

    filter = np.poly1d([1, 1])**sqares
    normaliser = 2**sqares

    print("filter ", filter)
    print("normaliser", normaliser);

    def binomcoeffs(n):
        return (np.poly1d([0.5, 0.5])**n).coeffs

    print("4: ", binomcoeffs(4))
    print("6: ", binomcoeffs(6))
    print("8: ", binomcoeffs(8))
    print("12: ", binomcoeffs(12))