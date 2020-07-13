import itertools
import numpy as np

def powerset(l):
    N = len(l)
    return [x for n in range(1, N + 1) for x in itertools.combinations(l, n)]

def checkSpecialPrime(sieve, num): 
      
    # While number is not equal to zero 
    while (num): 
          
        # If the number is not prime 
        # return false. 
        if (sieve[num] == False): 
            return False
  
        # Else remove the last digit 
        # by dividing the number by 10. 
        num = int(num / 10) 
  
    # If the number has become zero 
    # then the number is special prime, 
    # hence return true 
    return True
  
# Function to find the Smallest Special 
# Prime which is greater than or equal 
# to a given number 
def findSpecialPrime(N): 
    sieve = [True for i in range(N * 10 + 1)] 
  
    sieve[0] = False
    sieve[1] = False
  
    # sieve for finding the Primes 
    for i in range(2, N * 10 + 1): 
        if (sieve[i]): 
            for j in range(i * i, N * 10 + 1, i): 
                sieve[j] = False
      
    # There is always an answer possible 
    while (True): 
          
        # Checking if the number is a 
        # special prime or not 
        if (checkSpecialPrime(sieve, N)): 
              
            # If yes print the number 
            # and break the loop. 
            print(N) 
            break
      
        # Else increment the number. 
        else: 
            N += 1
  



PRIMES = np.array([2 ,3 ,5 ,7 ,11 ,13 ,17 ,19 ,23 ,29,
31 ,37 ,41 ,43 ,47 ,53 ,59 ,61 ,67 ,71,
73 ,79 ,83 ,89 ,97 ,101 ,103 ,107 ,109 ,113,
127 ,131 ,137 ,139 ,149 ,151 ,157 ,163 ,167 ,173,
179 ,181 ,191 ,193 ,197 ,199 ,211 ,223 ,227 ,229,
233 ,239 ,241 ,251 ,257 ,263 ,269 ,271 ,277 ,281,
283 ,293 ,307 ,311 ,313 ,317 ,331 ,337 ,347 ,349,
353 ,359 ,367 ,373 ,379 ,383 ,389 ,397 ,401 ,409,
419 ,421 ,431 ,433 ,439 ,443 ,449 ,457 ,461 ,463,
467 ,479 ,487 ,491 ,499 ,503 ,509 ,521 ,523 ,541,
547 ,557 ,563 ,569 ,571 ,577 ,587 ,593 ,599 ,601,
607 ,613 ,617 ,619 ,631 ,641 ,643 ,647 ,653 ,659,
661 ,673 ,677 ,683 ,691 ,701 ,709 ,719 ,727 ,733,
739 ,743 ,751 ,757 ,761 ,769 ,773 ,787 ,797 ,809,
811 ,821 ,823 ,827 ,829 ,839 ,853 ,857 ,859 ,863,
877 ,881 ,883 ,887 ,907 ,911 ,919 ,929 ,937 ,941,
947 ,953 ,967 ,971 ,977 ,983 ,991 ,997 ,1009 ,1013])

def smallest_prime(n):
    return int(PRIMES[PRIMES >= n].min())
