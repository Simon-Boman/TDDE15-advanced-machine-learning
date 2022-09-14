#################### LAB 2 #################### 
library(HMM)
set.seed("12345")


#################### ASSIGNMENT 1 #################### 
# Build a hidden Markov model (HMM) for the scenario described above

# Equal probability move or stay, i.e. 0.5.
# So state 1, probability go to state 2 or stay in state 1 is 0.5 for both etc..

prob_stay = 0.5
prob_move = 0.5

# The starting probabilities of the states
initial_probs = rep(0.1, 10)

# The transition probabilities between the states
transition_probs = diag(prob_stay, 10)
transition_probs[row(transition_probs) + 1 == col(transition_probs)] = prob_move
transition_probs[10,1] = 0.5

# The emission probabilities of the states
emission_probs = matrix(
  c(0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2,
    0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2,
    0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0,
    0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0,
    0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0,
    0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0,
    0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0,
    0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2,
    0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2,
    0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2
  ), nrow = 10, byrow = TRUE)

HMM_model = initHMM(States = c('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'), 
                    Symbols = c('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'), 
                    startProbs=initial_probs, transProbs=transition_probs, emissionProbs=emission_probs)
HMM_model



#################### ASSIGNMENT 2 #################### 
# Simulate the HMM for 100 time steps.
# Returns a path of states, and the sequence of our observations
# As given in the assignment, an observation of the true state has a 20% chance of being correct
HMM_simulations = simHMM(HMM_model, 100)
HMM_simulations


#################### ASSIGNMENT 3 #################### 
# Discard the hidden states from the sample obtained above. 
# Use the remaining observations to compute the filtered and smoothed probability distributions for each of the 100 time points. 
# Compute also the most probable path.

# Extract our observations
observations = HMM_simulations$observation
observations

# Compute our alpha and beta using the forward and backwards step of the forward-backward algorithm
alpha = prop.table(exp(forward(HMM_model, observations))) # the forward probabilities  
beta = prop.table(exp(backward(HMM_model, observations))) # the backward probabilities

# Since we have 10 states, and we simulated 100 time steps, we get a 10x100 matrix for the filtering and smoothing probability distribution
filtering_probs = matrix(0, 10, 100)
for (i in 1:100)
  # for each time step i, compute the filtering probability distribution (i.e. the probability for each of the 10 states)
  filtering_probs[,i] = alpha[,i] / sum(alpha[,i])
filtering_probs

smoothing_probs = matrix(0, 10, 100)
for (i in 1:100)
  smoothing_probs[,i] = alpha[,i] * beta[,i] / sum(alpha[,i] * beta[,i])
smoothing_probs


# The most probable path can be computed using the Viterbi algorithm
most_probable_path = viterbi(HMM_model, observations)
most_probable_path


#################### ASSIGNMENT 4 #################### 
# Compute the accuracy of the filtered and smoothed probability distributions, and of the most probable path. 
# That is, compute the percentage of the true hidden states that are guessed by each method.

# Finally, recall that you can compare two vectors A and B elementwise as A==B, 
# and that the function table will count the number of times that the different elements in a vector occur in the vector.

# Find the most probable states for our filtering and smoothing probability distributions. 
# apply: X specifies the data, MARGIN = 2 specifies column-wise apply, FUN specifies the function to use. 
most_prob_state_filtering = apply(X = filtering_probs, MARGIN = 2, FUN = which.max)
most_prob_state_filtering

most_prob_state_smoothing = apply(X = smoothing_probs, MARGIN = 2, FUN = which.max)
most_prob_state_smoothing

# Extract our actual states
states = strtoi(HMM_simulations$states)
states

acc_filtering = sum(most_prob_state_filtering == states) / length(states)
acc_filtering
acc_smoothing = sum(most_prob_state_smoothing == states) / length(states)
acc_smoothing
acc_path = sum(most_probable_path == states) / length(states)
acc_path

#################### ASSIGNMENT 5 #################### 
# Repeat the previous exercise with different simulated samples. 
# smoothed > filtered and veterbi!!!



# In general, the smoothed distributions should be more accurate than the filtered distributions. Why? 
# In general,the smoothed distributions should be more accurate than the most probable paths, too. Why ?




#################### ASSIGNMENT 6 #################### 
#################### ASSIGNMENT 7 #################### 

