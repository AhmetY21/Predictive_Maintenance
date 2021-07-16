import pandas as pd
import numpy as np
np.random.seed=0

def transition_matrix(chain,printed=False):
    n = 1+ max(chain)
    M = np.zeros((n,n))
    for (i,j) in zip(chain,chain[1:]):
        M[i][j] += 1
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    if printed:
        print(M)
    logging.info("Transition Matrix Calculated")
    return pd.DataFrame(M)


def Steady_State(state,transition,steps,visualise=True):
    stateHist=state
    for x in range(steps):
        state=np.dot(state,P)
        stateHist=np.append(stateHist,state,axis=0)
    if visualise:
        v=pd.DataFrame(stateHist)
        v.plot()
        plt.show()
    logging.info("Steady State Calculated")
    return state
