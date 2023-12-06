from collections import defaultdict
import string
import csv
import random
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm

def findWinner(row):
    if row["Home Points"] > row["Away Points"]:
        return row["Home Team"]
    else:
        return row["Away Team"]

def findLoser(row):
    if row["Home Points"] < row["Away Points"]:
        return row["Home Team"]
    else:
        return row["Away Team"]

def calcRowMargin(row):
    return abs(row["Home Points"] - row["Away Points"])
    
#This is our logitical regression which employs the help of the slope and intercept of our anticipated logistic regression to refine our results.
def log_r(margin, a, b):
    return 1 / (1 + np.exp(-a * margin - b))

def transitionMatrix(fileName,a,b,h):
    df = pd.read_csv(fileName)
    df = df.dropna(subset=['Home Points', 'Away Points'])
    allTeamNames = []
    for index, row in df.iterrows():
        winner = findWinner(row)
        loser = findLoser(row)
        if winner not in allTeamNames:
            allTeamNames.append(winner)
        if loser not in allTeamNames:
            allTeamNames.append(loser)
        print(winner, loser)

    Records = pd.DataFrame(0, index=allTeamNames, columns=["Wins", "Losses", "Distribution"])

    for index, row in df.iterrows():
        winner = findWinner(row)
        loser = findLoser(row)
        Records.at[winner, 'Wins'] += 1
        Records.at[loser, 'Losses'] += 1

    #This is our Markov Matrix
    M = pd.DataFrame(0, index=allTeamNames, columns=allTeamNames)

    # Update Markov matrix M, Populate the Markov transition matrix with probabilities based on game outcomes and margins (log_r)

    for index, row in df.iterrows():
        if row["Week"] < 16:
            nameW = findWinner(row)
            nameL = findLoser(row)
            HomeT = row["Home Team"]
            AwayT = row["Away Team"]
            marg = calcRowMargin(row)

            M.at[HomeT, HomeT] += 1
            M.at[AwayT, AwayT] += 1
            if row["Neutral Site"] == ["false"]:
                M.at[HomeT, AwayT] += 1 - log_r(marg, a, b)
                M.at[AwayT, HomeT] += log_r(marg, a, b)
            else:
                M.at[HomeT, AwayT] += 1 - log_r(marg + h, a, b)
                M.at[AwayT, HomeT] += log_r(marg + h, a, b)

    M.to_csv(f"Initial-Matrix.csv", index=False)

    #Normalize Transition Matrix such that each row sums to 1, which inturn calculates our probabilities.

    for team in M.columns:
        N = M.loc[team, team]
        M.loc[team] = M.loc[team] / N
        M.at[team, team] = 1 - M.loc[team].sum() + M.at[team, team]

    #print(M)

    M = M.fillna(0.0)

    M.to_csv(f"Normalized-Matrix.csv", index=False)

    #P becomes our stationary distribution as we multiply our transition matrix by itself and our initital markov transition matrix.
    P = M.copy()
    for i in range(400):
        P = P.dot(M)

    P.to_csv(f"Stationairy-Distribution.csv", index=False)

    #print(P)
    #print("_______________________________________________")
    #print(Records)

    # Update Records with stationary distribution
    for name in P.index:
        Records.at[name, 'Distribution'] = P.loc[name, name]

    Records.sort_values(by='Distribution', ascending=False, inplace=True)
    Records.reset_index(inplace=True)
    Records.rename(columns={'index': 'Team'}, inplace=True)

    # Write to CSV file
    Records.to_csv(f"RecordListSorted.csv", index=False)

    #return P

    return Records

def main():
    pd.set_option('display.max_columns', None)
    pd.options.display.max_rows
    #gen_rM()
    a = 0.0228092 
    b = -0.08489695
    h = 3.710938
    print(transitionMatrix('./Data/TrainingData.csv',a,b,h))



if __name__ == "__main__":
    main()


#a = 0.0228092
#b = -0.08489695 
#h = 3.710938