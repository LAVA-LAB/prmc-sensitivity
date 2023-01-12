import cvxpy as cp
import numpy as np

class poly(object):
    
    def __init__(self, param, coeff):
        '''
        Polynomial object
        
        Parameters
        ----------
        param : Parameter for this polynomial
        coeff : Dictionary or list for the coefficients
        '''
        
        if type(coeff) == dict:
            self.coeff = [0]*(max(coeff.keys())+1)
            for k,v in coeff.items():
                self.coeff[k] = v
        else:
	        self.coeff = coeff
                
        self.par  = param
    
    def __str__(self):
        print_list = []
        for p,c in enumerate(self.coeff):
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
                add_p = 'v'
            else:
                add_p = 'v^'+str(p)
                
            print_list += [add_c + add_p]
        
        return ' + '.join(print_list)
    
    def deriv_eval(self, param):
        # Differentiate polynomial and return the value evaluated at the 
        # current parameter value
        
        # Check if ID of the provided parameter equals that of this polynomial
        if param.id == self.par.id:
        
            coeff_der = np.polynomial.polynomial.polyder(self.coeff)
    
            return sum([c * self.par.value ** i for i,c in enumerate(coeff_der)])
    
        # If the IDs don't match, then the derivative is zero by default
        else:
            return 0
    
    def expr(self):
        # Evaluate the polynomial to get the CVXPY expression
        
        expr = 0
        for p,c in enumerate(self.coeff):
            if p == 0:
                expr += c
            elif p == 1:
                expr += self.par
            elif c == 0:
                continue
            elif c == 1:
                expr += self.par ** p
                
        return expr
        
        # return cp.sum([c * self.par ** i for i,c in enumerate(self.coeff)])
    
    def val(self):
        # Evaluate the polynomial and return the value evaluated at the 
        # current parameter value
        
        return cp.sum([c * self.par.value ** i for i,c in enumerate(self.coeff)])
    
    

class polytope(object):
    
    def __init__(self, A, b):
        
        self.A = A
        self.b = b
        
    def showA(self):
        
        string = np.vectorize(lambda x: x.print() if isinstance(x, poly) else str(x), 
                             otypes=[str])
        
        return string(self.A)
        
    def dA_eval(self):
        # Differentiate A matrix and return the value evaluated at the 
        # current parameter value
        
        dA = np.array([[
            
            ]])
        
        return 