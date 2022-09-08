#################### LAB 1 #################### 
library(bnlearn)
library(gRain)
data("asia")
set.seed("12345")



#################### ASSIGNMENT 1 #################### 
# Learn BN structure using hill climbing with default settings but changing number of restarts 
dag1 = hc(x = asia, restart = 1)
dag2 = hc(x = asia, restart = 5)
dag3 = hc(x = asia, restart = 50)
dag4 = hc(x = asia, restart = 200)
graphviz.plot(dag1, main = '1 restart')
graphviz.plot(dag2, main = '5 restarts')
graphviz.plot(dag3, main = '50 restarts')
graphviz.plot(dag4, main = '200 restarts')

# Can easily see that they are equal by checking number of edges, and if the BNs are equal:
nrow(arcs(dag1))
nrow(arcs(dag2))
nrow(arcs(dag3))
nrow(arcs(dag4))
all.equal(dag1, dag2)
all.equal(dag1, dag3)
all.equal(dag1, dag4)

# All produced DAGs are Markov Equivalent since they have the same edges (but some have changed direction)
# still equivalent BN structures. just change of orientations of some edges.


# Learn BN structures using hill climbing with default settings, 50 restarts, and different scoring mechanisms
dag5 = hc(x = asia, restart = 50, score = 'loglik')
dag6 = hc(x = asia, restart = 50, score = 'aic')
dag7 = hc(x = asia, restart = 50, score = 'k2')
dag8 = hc(x = asia, restart = 50, score = 'bic')
graphviz.plot(dag5, main = 'loglik')
graphviz.plot(dag6, main = 'aic')
graphviz.plot(dag7, main = "k2")
graphviz.plot(dag8, main = 'bic')

# Using different scoring mechanisms result in different DAGs that are in some cases NOT Markov Equivalent!
# We can see this easily when comparing number of edges:
nrow(arcs(dag5))
nrow(arcs(dag6))
nrow(arcs(dag7))
nrow(arcs(dag8))

# Or by checking if the BNs are equal:
all.equal(dag5, dag6)
all.equal(dag5, dag7)
all.equal(dag5, dag8)

# Since HC can get stuck in local optimas apart from changing scoring mechanism,
# also using different starting points (different BNs), could lead to different results. 



#################### ASSIGNMENT 2 #################### 
# Split the data into training and test set
n=dim(asia)[1]
set.seed(1337)
id=sample(1:n, floor(n*0.8))
traindata=asia[id,]
testdata=asia[-id,]

# Function for predicting S given a BN, data, and which nodes we have observed in the BN. 
predict_S = function(bn, data, observed_nodes) {
  # vector for storing predictions
  predictions = c(1:nrow(data))
  
  # for each of our observations, we have to update the observed values i.e. the evidence, 
  # on A, T, L, B, E, X, D - i.e. all nodes except S, since we want to predict S using our BN.
  # We can do this with setEvidence to update our grain object.
  # Thus for each training sample, we have to store each nodes observed state, 
  # and then use this to query our network in order to get a probability distribution over S. 
  for (i in 1:nrow(data)) {
    case =  subset(data[i,], select = observed_nodes)
    observed_states = NULL
    for (state in 1:ncol(case)) {
      if (case[state] == 'yes') {
        observed_states = c(observed_states, 'yes')
      }
      else {
        observed_states = c(observed_states, 'no')
      }
    }
    
    # Obtain probabilities over S by querying our grain network which we update with our observations
    # setEvidence - set, update or remove evidence of grain object,
    # i.e. we can update our grain object with new evidence. 
    # querygrain - Query an independence network, i.e. obtain the conditional distribution of a set of variables, 
    # typically given finding/evidence on other variables.
    probabilities = querygrain(object = setEvidence(bn, nodes = observed_nodes, states = observed_states), 
                               nodes = "S", 
                               type="joint")
    
    if (probabilities['yes'] > 0.5) {predictions[i] = 'yes'} else {predictions[i] = 'no'}
  }
  return(predictions)
}

# Learn the structure of our BN
trained_dag = hc(x = traindata)
graphviz.plot(trained_dag, main = 'trained_dag')

# Fit the parameters of our BN, this gives us all the conditional probability 
# tables for all our parameters i.e. their probabilities of yes/no
fitted = bn.fit(x = trained_dag, traindata)
fitted

# Convert the BN to a gRain object - Graphical Independence Network.
# Grain is an implementation of exact inference in R for discrete BNs. 
# The gRain class object stores a fitted BN as a list of conditional probability tables. 
grain = as.grain(fitted)
# Compiled: TRUE, objects are compiled upon creation by default.- 
# Compiling means creating a junction tree and establishing clique potentials
grain
# Gives us the cliques, separates, and nr of parents for each node. 
grain$rip
# Plots the gRain graph. 
plot(grain)

# Predictions on S from testdata using above function and all nodes except of S
predictions = predict_S(grain, testdata, c("A", "T", "L", "B", "E", "X", "D"))

# CM and accuracy
cm = table(predictions, testdata$S)
cm
acc_cm = (cm[1]+cm[4])/sum(cm)
acc_cm


# Compare result with the true Asia BN:
true_dag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
graphviz.plot(true_dag, main = 'true DAG')
fitted_true = bn.fit( x = true_dag, traindata)
grain_true = as.grain(fitted_true)
grain_true
grain_true$rip
plot(grain_true)

predictions_true = predict_S(grain_true, testdata, c("A", "T", "L", "B", "E", "X", "D"))

cm_true = table(predictions_true, testdata$S)
cm_true
acc_true = (cm_true[1]+cm_true[4])/sum(cm_true)
acc_true



#################### ASSIGNMENT 3 #################### 
# Get the markov blanket from the fitted BN, 
# these are the observed nodes we will use for classifying S
markov_blanket = mb(fitted, 'S')
markov_blanket

# Predict using only the markov blanket
predictions_mb = predict_S(grain, testdata, markov_blanket)
cm_mb = table(predictions_mb, testdata$S)
cm_mb
acc_mb = (cm_mb[1]+cm_mb[4])/sum(cm_mb)
acc_mb

# Compare result with the true Asia BN:
markov_blanked_true = mb(fitted_true, 'S')
markov_blanked_true
predictions_mb_true = predict_S(grain_true, testdata, markov_blanked_true)
cm_mb_true = table(predictions_mb_true, testdata$S)
cm_mb_true
acc_mb_true = (cm_mb_true[1]+cm_mb_true[4])/sum(cm_mb_true)
acc_mb_true



#################### ASSIGNMENT 4 #################### 
# Naive Bayes classifier - predictive variables are independent given the class variable S!
# Thus we should have a graph where S is connected with an edge to all other variables, and nothing else. 

graphviz.plot(model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]"), main = 'true DAG')
graphviz.plot(trained_dag, main = 'true DAG') 

naive_dag = model2network("[S][A|S][L|S][T|S][E|S][B|S][X|S][D|S]")
graphviz.plot(naive_dag, main = 'naive DAG')

fitted_naive = bn.fit( x = naive_dag, traindata)
grain_naive = as.grain(fitted_naive)
grain_naive
grain_naive$rip
plot(grain_naive)

predictions_naive = predict_S(grain_naive, testdata, c("A", "T", "L", "B", "E", "X", "D"))
cm_naive = table(predictions_naive, testdata$S)
cm_naive
acc_naive = (cm_naive[1]+cm_naive[4])/sum(cm_naive)
acc_naive