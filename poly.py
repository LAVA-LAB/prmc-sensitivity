import cvxpy as cp
import numpy.polynomial.polynomial as poly

class pol(object):
    
    def __init__(self, param, coeff):
        
        self.coeff = coeff
        self.par  = param
    
    def deriv_eval(self, param):
        # Differentiate polynomial and return the value evaluated at the 
        # current parameter value
        
        # Check if ID of the provided parameter equals that of this polynomial
        if param.id == self.par.id:
        
            coeff_der = poly.polyder(self.coeff)
    
            return cp.sum([c * self.par ** i for i,c in enumerate(coeff_der)]).value
    
        # If the IDs don't match, then the derivative is zero by default
        else:
            return 0
    
    def expr(self):
        # Evaluate the polynomial to get the CVXPY expression
        
        return cp.sum([c * self.par ** i for i,c in enumerate(self.coeff)])
    
    def val(self):
        # Evaluate the polynomial and return the value evaluated at the 
        # current parameter value
        
        return cp.sum([c * self.par.value ** i for i,c in enumerate(self.coeff)])