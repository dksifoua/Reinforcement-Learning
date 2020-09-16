# Decision making under uncertainty

## Sequential decision making with evaluative feedback

In RL, the agent generates its own training data by interacting with the world. The agent must learn the consequences 
of its own actions through trials and errors rather than being told the correct actions.

A. **The K-armed bandit problem**

In a k-armed bandit problem, we have an **agent** who chooses between <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> **actions** and receives a **reward** based 
on the action it chooses.

B. **Action-Values**

For the agent decides which action is best, we must define a velue of taking each action called 
**action-values function**.

- The **value** is the **expected reward** the agent receives when taking an action.

<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/6744a8c20f7ca77cfc538d489573f5bd.svg?invert_in_darkmode&sanitize=true" align=middle width=362.05104209999996pt height=36.16460595pt/></p>

- The goal of the agent is to **maximize** the **expected reward** by selecting the action that have the highest value.

<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/7d24046cd50886679a4deef092cfbb4f.svg?invert_in_darkmode&sanitize=true" align=middle width=106.3980918pt height=16.438356pt/></p>

## Learning Action-Values

The objective is to **estimate** <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/a1ade33520272d5639a74a65b579e137.svg?invert_in_darkmode&sanitize=true" align=middle width=96.41060054999998pt height=24.65753399999998pt/>.

A. **Sample-Average Method**

One way to estimate <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/899d6b817846e664545e243fb0fbfcbf.svg?invert_in_darkmode&sanitize=true" align=middle width=36.36998804999999pt height=24.65753399999998pt/> is to compute a sample average:

<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/76de8a18243410c3b9fde977e970a3eb.svg?invert_in_darkmode&sanitize=true" align=middle width=195.43062329999998pt height=45.82666275pt/></p>

If the denominator is <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>, we defined <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/bd1e1ceb5a0e32e724c780a60015d05f.svg?invert_in_darkmode&sanitize=true" align=middle width=34.47001139999999pt height=24.65753399999998pt/> at some default value such as <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>.

As <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/5d02ef7b9f617aadde96b19313c32602.svg?invert_in_darkmode&sanitize=true" align=middle width=168.12286755pt height=31.36100879999999pt/>, <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/0dbec9a907b25f0be3daa33fd54f0cd0.svg?invert_in_darkmode&sanitize=true" align=middle width=102.1982874pt height=24.65753399999998pt/> (*Law of large numbers*).

The sample-average method is not necessarily the best one for selecting action-values.

Calculating <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/03e5774df2134414fe1121641a9de435.svg?invert_in_darkmode&sanitize=true" align=middle width=40.257699899999984pt height=24.65753399999998pt/> by using the formula above implies memory and computational requirements which increase as <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> 
increases. So, we  use an **incremental implementation**

Let's <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/d61bbb1e022c6bd27837bf8fd147cff9.svg?invert_in_darkmode&sanitize=true" align=middle width=152.9192115pt height=31.945230899999984pt/>

<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/2de6031df57e384d198a2e9ded3ef4b4.svg?invert_in_darkmode&sanitize=true" align=middle width=913.81392465pt height=47.35857885pt/></p>

We obtain the update rule: **New Estimate <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/a14d504f11ac9590eea24397c59fab71.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> Old Estimate + Step Size [Target - Old Estimate]**

Step Size = <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/b548eadabf0a95bf9fa76e55d89f7784.svg?invert_in_darkmode&sanitize=true" align=middle width=159.05527769999998pt height=27.77565449999998pt/>

B. **Tracking non-stationary problem - Exponential recency-weighted average**

When the probability of reward changes over time (in most RL problems), it makes sense to give more weight to recent 
rewards than to long-past rewards.

<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/1c77b1a6ee7c1a5f5f9535cb7f001973.svg?invert_in_darkmode&sanitize=true" align=middle width=1631.0064441pt height=18.312383099999998pt/></p>
<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/082d2965b1c252e2237a87e3bbf4339d.svg?invert_in_darkmode&sanitize=true" align=middle width=312.32876895pt height=44.89738935pt/></p>

The convergence is guarantedd with <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/83f7e47ae689413e41bec72690810abc.svg?invert_in_darkmode&sanitize=true" align=middle width=72.95442329999999pt height=27.77565449999998pt/>.

Sometimes, it is convenient to change <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/329c0def892ca4d38851e6811d1e0961.svg?invert_in_darkmode&sanitize=true" align=middle width=40.93817144999999pt height=24.65753399999998pt/> from step to step. But of course convergence is guaranteed for all 
choices of the sequence <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/33e3e77301283a79cb8e69ecef3eb45b.svg?invert_in_darkmode&sanitize=true" align=middle width=57.376590149999984pt height=24.65753399999998pt/>.

A well-known result in stochastic approximation theory gives us the conditions required to ensure convergence with 
probability 1:

1. <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/4d44e861e3906a013971a15e4a09b4e6.svg?invert_in_darkmode&sanitize=true" align=middle width=124.97736404999998pt height=26.438629799999987pt/> guaranteed that the steps are large enough to eventually overcome any init 
conditions or random fluctuations.
2. <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/6a9c82eb347e4e0d6cb9538175d7aebc.svg?invert_in_darkmode&sanitize=true" align=middle width=124.97736404999998pt height=26.438629799999987pt/> guaranteed that eventually the steps become small enough to ensure 
convergence.

These conditions are **true** with <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/83f7e47ae689413e41bec72690810abc.svg?invert_in_darkmode&sanitize=true" align=middle width=72.95442329999999pt height=27.77565449999998pt/> but **false** for the case of constant stepsize param 
<img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> (which is desirable in non-stationary environments).

C. **Action selection**

- **Greedy action selection** (low reward variance): the agent always exploits its current knowledge to maximize the 
immediate reward.

<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/0b7e12882cb80ff8abd261db2d9c0f52.svg?invert_in_darkmode&sanitize=true" align=middle width=150.31992029999998pt height=16.438356pt/></p>

- **<img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/9ae7733dac2b7b4470696ed36239b676.svg?invert_in_darkmode&sanitize=true" align=middle width=7.66550399999999pt height=14.15524440000002pt/>-greedy action selection** (high reward variance): every once in a while, the agent an action with a 
small probability <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/9ae7733dac2b7b4470696ed36239b676.svg?invert_in_darkmode&sanitize=true" align=middle width=7.66550399999999pt height=14.15524440000002pt/>. Every action will be sampled an infinite number of time as the number of steps 
increase. Thus, <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/8e99045e915d771ec08416a3405f8a20.svg?invert_in_darkmode&sanitize=true" align=middle width=231.18384135pt height=24.65753399999998pt/>.

If the reward function is non-stationary (doesn't change over time), the exploration is needed i.e <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/9ae7733dac2b7b4470696ed36239b676.svg?invert_in_darkmode&sanitize=true" align=middle width=7.66550399999999pt height=14.15524440000002pt/>-greedy 
action selection.

## Exploration vs Exploitation tradeoff

The tradeoff is simply the way the agent decides when its takes the best action (according to its current knowledge) - 
exploitation or try something else (random) exploration. If the agent never chooses a particular action, it won't know 
its value.

- **Exploration**: *improve* knowledge for *long-term* benefit.

- **Exploitation**: *exploit* knowledge for *short-term* benefit.

When we explore, we get more accurate estimate of our values, when we exploit, we might get more reward. We cannot 
however choose to do both simultaneously.

A simple method to choose between exploration and exploitation is to use **<img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/9ae7733dac2b7b4470696ed36239b676.svg?invert_in_darkmode&sanitize=true" align=middle width=7.66550399999999pt height=14.15524440000002pt/>-greedy action selection**. 
Other methods are **optimistic initial values** and **Upper-Confidence Bound (UCB) action selection**.

A. **Optimistic initial values**

All methods discussed so far depend on init value <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/7aea71830b8148af629660970435dff3.svg?invert_in_darkmode&sanitize=true" align=middle width=41.84447189999999pt height=24.65753399999998pt/>. These methods are **biased** since we use the statistic 
mean. The bias disappears with sampled average and is constant with fixed value of <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>.

In practice, it's not usually a problem and can sometimes be very helpfull.

B. **Limitations of optimistic initial values**

- Only drives early exploration.
- Not well-suited for non-stationary problems.
- Sometimes, it's difficult to know what the optimistic initial values should be. Then, we have to tune them.

C. **Upper-Confidence Bound (UCB) action selection**

UCB action selection uses **uncertainty** in the value estimates for balancing exploration and exploitation.

In other words, it would be better to select among non-greedy actions according to their potential for being actually 
optimal taking into account:

- How close the estimates are being max.
- The uncertainties in those estimates.

One effetive way to do that is:

<p align="center"><img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/1d9727ff2cd1473d02dd7fe33ecf56c4.svg?invert_in_darkmode&sanitize=true" align=middle width=242.94483839999998pt height=49.315569599999996pt/></p>

Where <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/473d9e806b252cce05095245d5c760cc.svg?invert_in_darkmode&sanitize=true" align=middle width=40.47004334999999pt height=24.65753399999998pt/> is the number of time the action <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> is selected at time step <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> and <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode&sanitize=true" align=middle width=7.11380504999999pt height=14.15524440000002pt/> control the exploration.

UCB action selection always performs well than <img src="/notebooks/01_Decision_Making_Under_Uncertainty/tex/9ae7733dac2b7b4470696ed36239b676.svg?invert_in_darkmode&sanitize=true" align=middle width=7.66550399999999pt height=14.15524440000002pt/>-greedy action selection but it's **more difficult** to 
extend to more general RL problems:

- Deal with non-stationary problems.
- Deal with large state spaces particularly using function approximation.

In these more advanced settings, UCB action selection is usually not practical.