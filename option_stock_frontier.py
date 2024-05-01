import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats


def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# weight is related to the stock position, so option weight is 1 - weight
def portfolio(weight, option_price):
    print('Price of Option:', option_price)

    call = lambda x: max(x - S, 0)
    short_call = lambda x: max(S - x, 0)

    prices = np.linspace(0, 2 * S, 1000)
    payouts = np.zeros(len(prices))

    num_options = (100 - weight) / option_price

    print('Number of Options:', num_options)

    i = 0
    for price in prices:
        if weight <= 1000:
            payouts[i] = ((weight / 100) * (price - S)) + (num_options * call(price))
        else:
            payouts[i] = (weight / 100) * (price - S) + num_options * short_call(price)
        i += 1

    payouts = payouts / S

    return prices, payouts


def plot_portfolio(prices, payouts, weight):
    plt.plot(prices, payouts * 100)
    plt.xlabel('Stock Price')
    plt.ylabel('Portfolio Value (% of Initial Value)')
    plt.title(f'Portfolio Value for Stock Weight of {weight}%')
    plt.show()

    return


def payout_distribution(prices, payouts):
    mu = 100
    stddev = 10
    # x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    probs = stats.norm.pdf(prices, mu, stddev)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(prices, probs, color='red')
    ax2.plot(prices, payouts)

    plt.show()

    return


def CalculateCVaR(prices, payouts):
    CVaR = 0
    prob = 0

    mu = 100
    stddev = 10

    probs = stats.norm.pdf(prices, mu, stddev)

    i = 0

    while prob < 0.05:
        CVaR += probs[i] * payouts[i]
        prob += probs[i] / 5
        i += 1

    print('CVaR (5%):', CVaR / 5)

    return -CVaR / 5


def CalculateExpectedReturn(e_r, weight, num_options):
    return (e_r * (weight / 100)) + (e_r * num_options)


if __name__ == '__main__':
    w = 0
    S = 100
    K = 100
    T = 1
    r = 0.03
    sigma = 0.2
    option_type = 'call'
    option_price = black_scholes(S, K, T, r, sigma, option_type)

    e_r = 0.15

    prices, payouts = portfolio(w, option_price)

    #plot_portfolio(prices, payouts, w)
    #payout_distribution(prices, payouts)


    weights = np.linspace(0, 100, 101)
    
    CVaRs = []
    er_s = []

    for weight in weights:
        prices, payouts = portfolio(weight, option_price)
        CVaRs.append(CalculateCVaR(prices, payouts))

        num_options = (100 - weight) / option_price
        er_s.append(CalculateExpectedReturn(e_r, weight, num_options))



    plt.plot(CVaRs, er_s)
    plt.xlabel('CVaR')
    plt.ylabel('Expected Return')
    plt.title('Stock Weight vs CVaR')
    plt.show()
