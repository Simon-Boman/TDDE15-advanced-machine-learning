#################### LAB 2 #################### 
library(HMM)
library(enropy)
set.seed("12345")



#################### ASSIGNMENT 1 #################### 
# Equal probability to move to next state or stay in current state, i.e. 0.5.
prob_stay = 0.5
prob_move = 0.5

# The starting probabilities of the states is assumed to be equal
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



#################### ASSIGNMENT 2 #################### 
# Simulates the HMM for 100 time steps, returns a path of states, and the sequence of our observations.
HMM_simulations = simHMM(HMM_model, 100)

# As given in the assignment, an observation of the true state has a 20% chance of being correct
sum(HMM_simulations$states == HMM_simulations$observation)



#################### ASSIGNMENT 3 #################### 
# Extract our observations
observations = HMM_simulations$observation

# Compute alpha in the forward step of the forward-backward algorithm, 
# which we need to compute the filtering probabiliy distribution (i.e. the probability for each of the 10 states for each of the 100 time steps)
filtering_probs = prop.table(exp(forward(HMM_model, observations)))
for (i in 1:100) {filtering_probs[,i] = filtering_probs[,i] / sum(filtering_probs[,i])}

# Compute alpha and beta in the forward and backward step of the forward-backward algorithm, 
# which we need to compute the smoothing probabiliy distribution (i.e. the probability for each of the 10 states for each of the 100 time steps)
smoothing_probs = prop.table(exp(posterior(HMM_model, observations)))
for (i in 1:100) {smoothing_probs[,i] = smoothing_probs[,i] / sum(smoothing_probs[,i])}

# Since we have 10 states, and we simulated 100 time steps, we get a 10x100 matrix for the filtering and smoothing probability distribution

# The most probable path can be computed using the Viterbi algorithm
most_probable_path = viterbi(HMM_model, observations)



# Alternative way of computing filtering and smoothing probabilities
# Compute our alpha and beta using the forward and backwards step of the forward-backward algorithm
alpha = prop.table(exp(forward(HMM_model, observations))) # the forward probabilities  
beta = prop.table(exp(backward(HMM_model, observations))) # the backward probabilities
filtering_probs = matrix(0, 10, 100)
for (i in 1:100) {filtering_probs[,i] = alpha[,i] / sum(alpha[,i])}
smoothing_probs = matrix(0, 10, 100)
for (i in 1:100) {smoothing_probs[,i] = alpha[,i] * beta[,i] / sum(alpha[,i] * beta[,i])}



#################### ASSIGNMENT 4 #################### 
# Find the most probable states for our filtering and smoothing probability distributions. 
# apply: X specifies the data, MARGIN = 2 specifies column-wise apply, FUN specifies the function to use. 
most_prob_state_filtering = apply(X = filtering_probs, MARGIN = 2, FUN = which.max)
most_prob_state_smoothing = apply(X = smoothing_probs, MARGIN = 2, FUN = which.max)

# Extract our actual states
states = strtoi(HMM_simulations$states)
states

# Compute accuracies of the filtering distribution, smoothing distribution, and predicted path
acc_filtering = sum(most_prob_state_filtering == states) / length(states)
acc_smoothing = sum(most_prob_state_smoothing == states) / length(states)
acc_path = sum(most_probable_path == states) / length(states)
acc_filtering
acc_smoothing
acc_path



#################### ASSIGNMENT 5 #################### 
# Function that takes our model and n time steps, and simulates new states and observations for the given model.
# The accuracy for using filtering, smoothing, and most probable path are then computed and returned. 
HMM_accuracies = function(model, n) {
  HMM_simulations = simHMM(model, n)
  observations = HMM_simulations$observation

  # Compute alpha in the forward step of the forward-backward algorithm, 
  # which we need to compute the filtering probabiliy distribution (i.e. the probability for each of the 10 states for each of the 100 time steps)
  filtering_probs = prop.table(exp(forward(model, observations)))
  for (i in 1:n) {filtering_probs[,i] = filtering_probs[,i] / sum(filtering_probs[,i])}
  
  # Compute alpha and beta in the forward and backward step of the forward-backward algorithm, 
  # which we need to compute the smoothing probabiliy distribution (i.e. the probability for each of the 10 states for each of the 100 time steps)
  smoothing_probs = prop.table(exp(posterior(model, observations)))
  for (i in 1:n) {smoothing_probs[,i] = smoothing_probs[,i] / sum(smoothing_probs[,i])}
  
  most_probable_path = viterbi(model, observations)
  
  most_prob_state_filtering = apply(X = filtering_probs, MARGIN = 2, FUN = which.max)
  most_prob_state_smoothing = apply(X = smoothing_probs, MARGIN = 2, FUN = which.max)
  
  states = strtoi(HMM_simulations$states)
  
  # Compute accuracies of the filtering probability distribution, smoothing probability distribution, and predicted path
  acc_filtering= sum(most_prob_state_filtering == states) / length(states)
  acc_smoothing = sum(most_prob_state_smoothing == states) / length(states)
  acc_path = sum(most_probable_path == states) / length(states)
  
  return(c(acc_filtering, acc_smoothing, acc_path))
}

# Compute accuracies for some new samples
new_samples = 10
accuracies = matrix(0, nrow = new_samples, ncol = 3)
colnames(accuracies) = c('filtering', 'smoothing', 'path')
for (i in 1:new_samples) {
  sample = HMM_accuracies(HMM_model, 100)
  accuracies[i,] = sample
}

means = matrix(0, nrow = 1, ncol = 3)
colnames(means) = c('filtering', 'smoothing', 'path')
means[1,] = c(mean(accuracies[,1]), mean(accuracies[,2]), mean(accuracies[,3]))

accuracies
means



#################### ASSIGNMENT 6 #################### 
# We want entropy close to 0, since that means we are more secure about our distribution. 
# High entropy mean we are unsure about our distribution, if e.g. equal probability. 
# entropy_empirical[96] = 0, if we look filtering_probs[96] we see it has probability = 1 for state 1, probability 0 for rest of the states, thus very sure.
# entropy_empirical[1] = 1.5+, if we look at filtering_probs[1] we see it has probability = 0.2 for 5 states, thus very unsure. 
entropy_empirical_filtering = apply(filtering_probs, MARGIN = 2, FUN=entropy.empirical)
plot(entropy_empirical_filtering, type='l')

# From this plot, appears the entropy decreases as t increases, 
# since there are more high peaks in the start, and the entropy is close to 0 more often towards the end. 
# But it is not very apparent, and should be investigated further.

# Means from previous exercise where we did multiple simulations using t = 100
means 

accuracies_200_steps = HMM_accuracies(HMM_model, 200)
accuracies_200_steps

accuracies_300_steps = HMM_accuracies(HMM_model, 300)
accuracies_300_steps


# Compute mean accuracies of 100 simulations using t=100
new_samples = 100
accuracies = matrix(0, nrow = new_samples, ncol = 3)
colnames(accuracies) = c('filtering', 'smoothing', 'path')
for (i in 1:new_samples) {accuracies[i,] = HMM_accuracies(HMM_model, 100)}
means = matrix(0, nrow = 1, ncol = 3)
colnames(means) = c('filtering', 'smoothing', 'path')
means[1,] = c(mean(accuracies[,1]), mean(accuracies[,2]), mean(accuracies[,3]))
means

# Compute mean accuracies of 100 simulations using t=300
accuracies_t300 = matrix(0, nrow = new_samples, ncol = 3)
colnames(accuracies_t300) = c('filtering', 'smoothing', 'path')
for (i in 1:new_samples) {accuracies_t300[i,] = HMM_accuracies(HMM_model, 100)}
means_t300 = matrix(0, nrow = 1, ncol = 3)
colnames(means_t300) = c('filtering', 'smoothing', 'path')
means_t300[1,] = c(mean(accuracies_t300[,1]), mean(accuracies_t300[,2]), mean(accuracies_t300[,3]))
means_t300



#################### ASSIGNMENT 7 #################### 
smoothing_probs[,100]
# Since we are looking at the last time step, t = 100, using filtering here is equal to using smoothing. 
filtering_probs[,100]

# We defined our transition probabilities at the beginning, and we can use these to 
# compute the probability of being in each state at time step t = 101, 
# given that we have the probabilities of being in each state at time t = 100

# Note: we use the transition probabilities and not the emission probabilities, cause we are interested in the true state,
# not which state we will actually observe. 
smoothing_probs[,100] %*% transition_probs

