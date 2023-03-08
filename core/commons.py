import time
import numpy as np
from scipy import sparse


from core.polynomial import polynomial

def TicTocGenerator():
    ''' Generator that returns the elapsed run time '''
    ti = time.time() # initial time
    tf = time.time() # final time
    while True:
        tf = time.time()
        yield tf-ti # returns the time difference



def TicTocDifference():
    ''' Generator that returns time differences '''
    tf0 = time.time() # initial time
    tf = time.time() # final time
    while True:
        tf0 = tf
        tf = time.time()
        yield tf-tf0 # returns the time difference



TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
TicTocDiff = TicTocDifference() # create an instance of the TicTocGen generator



def toc(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds." %tempTimeInterval )



def tic():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    


def tocDiff(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicTocDiff)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %np.round(tempTimeInterval, 5) )
    else:
        return np.round(tempTimeInterval, 12)
        
    return tempTimeInterval



def ticDiff():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    tocDiff(False)
    
    
    
def unit_vector(size, pos):
    v = np.zeros(size)
    v[pos] = 1
    return v



valuate = np.vectorize(lambda x: x.val() if isinstance(x, polynomial) else x, 
                     otypes=[float])

deriv_valuate = np.vectorize(lambda x,y: x.deriv_eval(y) if isinstance(x, polynomial) else 0, 
                     otypes=[float])



def rrange(start, length):
    
    return np.arange(start, start+length)



def expected_visits(J, sI):
    '''
    Get the expected number of visits of every state in a Markov chain
    '''
    
    S = sparse.linalg.spsolve(J, sparse.identity(J.shape[0]))
    S_arr = S.toarray()
    
    return sI['p'] @ S_arr[sI['s']]



def get_state_distribution(pmc, state):
    '''
    Get the probability distribution of `state` in a Markov chain under the 
    policy loaded in `pmc`.
    '''
    
    a = np.where(pmc.scheduler_prob[state] == 1)[0]
    sa = pmc.model.states[state].actions[a]
    
    prob = [t.value() for t in sa.transitions]
    succ = [t.column for t in sa.transitions]
    
    return prob, succ