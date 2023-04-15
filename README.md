# Optimal Matrix Chain Ordering Algorithm

This repository implements the $O(n \log n)$ algorithm for optimal
ordering of a matrix chain first developed by Hu and Shing, with heavy
use of Ramanan's simpler exposition of the same algorithm. 

Compared to the dynamic programming algorithm used by `numpy` and most
other libraries, which is $O(n^3)$, this algorithm is significantly
faster.

### Sources
Hu and Shing's work:

 - [Part 1](https://cse.hkust.edu.hk/mjg_lib/bibs/DPSu/DPSu.Files/0211028.pdf)
 - [Part 2](https://cse.hkust.edu.hk/mjg_lib/bibs/DPSu/DPSu.Files/0213017.pdf)
 
Ramanan's work:

[DOI link](https://epubs.siam.org/doi/10.1137/S0097539790190077)