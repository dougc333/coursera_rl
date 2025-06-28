prediction and control

policy evaluation: DP solves both policy evaluation and policy control problems
DP uses bellman equations to define iterative algos for policy eval and control
how to evaluate a policy? 
Calculate $v_{\pi}$. 

What is $v_{\pi}$?? It is the expected return 
$v_{\pi}(s) \doteq E_{pi}[G_t | S_t = s] $ 
<br/>
$v_{\pi}(s) = \sum\limits_{a} \pi(a|s) \sum\limits_{s'} \sum\limits_{r} p(s',r|s,a) [r+\gammav_{\pi}(s')]$
but we know $\sum \pi(a|s) = 1 $ for deterministic policies. 

Evaluation, can solve with a linear solver exactly or use dynamic programming for approximate solution

Control is process of improving policy. 

Policy2 is considered better than Policy1 if P2 is greater to or equal to P1 in every state. Strictly better is an important condition to prevent oscillations. If we have 2 policies which are equal we can oscillate between the 2. 2 equal policies, noe is never the winner. have to break a tie deterministically. 

Control is finding a policy which is strictly better than current policy. 

When you cant find a strictly better policy then that is the optimal policy. 


