import numpy as np

def sample(pdf, nbr_samples, n_max=10**6):
    np.random.seed(45)
    """ genearte a list of random samples from a given pdf
    suggests random samples between 0 and L 
    and accepts-rejects the suggestion with probability pdf(x) 
    """
    samples=[]
    n=0
    while len(samples)<nbr_samples and n<n_max:
        x=np.random.uniform(low=0,high=L)
        new_sample=pdf(x)
        assert new_sample>=0 and new_sample<=1
        if np.random.uniform(low=0,high=1) <=new_sample:
            samples += [x]
        n+=1
    return sorted(samples)


""" test """
L=1.0
rho_a=0.05; rho_b=0.95; gama=0.1
def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

pos0=sample(rho_int,20)
