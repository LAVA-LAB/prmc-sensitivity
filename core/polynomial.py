import cvxpy as cp
import numpy as np

class polynomial(object):
    
    def __init__(self, param, coeff, power):
        '''
        Polynomial object
        
        Parameters
        ----------
        param : Parameter for this polynomial
        coeff : Multiplication factors
        powers : Powers of the polynomials (noninteger allowed)
        '''
        
        self.par = param
        self.coeff = coeff
        self.power = power                
        self.par  = param
    
    def __str__(self):
        print_list = []
        for p,c in zip(self.power, self.coeff):
            if c == 0:
                continue
            elif c == 1 and p==0:
                add_c = '1'
            elif c == 1:
                add_c = ''
            else:
                add_c = str(c)
            
            if p == 0:
                add_p = ''
            elif p == 1:
                add_p = str(self.par)
            else:
                add_p = '*{}^{}'.format(self.par, p)
                
            print_list += [add_c + add_p]
        
        return ' + '.join(print_list)
    
    
    
    def deriv_eval(self, param):
        # Differentiate polynomial and return the value evaluated at the 
        # current parameter value
        
        # Check if ID of the provided parameter equals that of this polynomial
        if param.id == self.par.id:
        
            return sum([c * p * self.par.value ** (p-1) if p != 0 else 0 for p,c in zip(self.power, self.coeff)])    
    
        # If the IDs don't match, then the derivative is zero by default
        else:
            return 0
    
    
    
    def expr(self):
        # Evaluate the polynomial to get the CVXPY expression
        
        expr = 0
        for p,c in zip(self.power, self.coeff):
            if p == 0:
                expr += c
            elif p == 1:
                expr += self.par
            elif c == 0:
                continue
            elif c == 1:
                expr += self.par ** p
                
        return expr
    
    
    
    def val(self):
        # Evaluate the polynomial and return the value evaluated at the 
        # current parameter value
        
        val = cp.sum([c * self.par.value ** p for p,c in zip(self.power, self.coeff)])
        
        return val