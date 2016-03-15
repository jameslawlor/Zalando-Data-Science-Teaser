# zalando_datascience_teaser

Zalando are a European online fashion retailer and they set this challenge
to applicants for their Data Science jobs. The goal is to locate the whereabouts
of a new analyst that they wish to recruit, given some limited information about
her.

The solution involves modelling three probability distributions over a Cartesian
plane that represents Berlin. By combining these distributions and finding 
the maxima it's possible to figure out the most likely location(s).

I've given a run through of this project on my blog lawlorino.wordpress.com 

### Challenge Brief

The Zalando Data Intelligence Team is searching for a new top analyst. 
We already know of an excellent candidate with top analytical and programming
skills. Unfortunately, we don't know her exact whereabouts but we only 
have some vague information where she might be. Can you tell us where to best
send our recruiters and plot an easy to read map of your solution for them?

This is what we could extract from independent sources:

The candidate is likely to be close to the river Spree.
The probability at any point is given by a Gaussian function of its shortest 
distance to the river. The function peaks at zero and has 95% of its total
integral within +/-2730m
A probability distribution centered around the Brandenburg Gate also informs 
us of the candidate's location. The distribution's radial profile is log-normal
with a mean of 4700m and a mode of 3877m in every direction.
A satellite offers further information: with 95% probability she is located 
within 2400 m distance of the satellite's path (assuming a normal probability 
distribution)

Please make use of the additional information in the file http://bit.ly/19fdgVa.
