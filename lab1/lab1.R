#################### ASSIGNMENT 1 #################### 

library(bnlearn)
library(gRain)

data("asia")

set.seed("12345")
dag1 = hc(x = asia, restart = 1)
dag2 = hc(x = asia, restart = 5)
dag3 = hc(x = asia, restart = 50)
dag4 = hc(x = asia, restart = 200)
graphviz.plot(dag1, main = 'dag1')
graphviz.plot(dag2, main = 'dag2')
graphviz.plot(dag3, main = 'dag3')
graphviz.plot(dag4, main = 'dag4')

nrow(arcs(dag1))
nrow(arcs(dag2))
nrow(arcs(dag3))
nrow(arcs(dag4))
all.equal(dag1, dag2)
all.equal(dag1, dag3)
all.equal(dag1, dag4)


# still equivalent BN structures. just change of orientations of some edges.

dag5 = hc(x = asia, restart = 50, score = 'loglik')
dag6 = hc(x = asia, restart = 50, score = 'aic')
dag7 = hc(x = asia, restart = 50, score = 'bic')
dag8 = hc(x = asia, restart = 50, score = 'ebic')
graphviz.plot(dag5, main = 'dag5')
graphviz.plot(dag6, main = 'dag6')
graphviz.plot(dag7, main = 'dag7')
graphviz.plot(dag8, main = "dag8")

# using different scores we get very different DAGs!

# can see this easily when comparing number of edges
nrow(arcs(dag5))
nrow(arcs(dag6))
nrow(arcs(dag7))
nrow(arcs(dag8))

# or by checking if the BNs are equal:
# "Different number of directed/undirected arcs"
all.equal(dag5, dag6)
all.equal(dag5, dag7)
all.equal(dag5, dag8)


# since HC can get stuck in LOCAL optima, also using different starting points (different BNs), could lead to different results. 

#################### ASSIGNMENT 2 #################### 
n=dim(asia)[1]
set.seed(1337)
id=sample(1:n, floor(n*0.8))
traindata=asia[id,]
testdata=asia[-id,]

# Learn both structure and parameters. Use any learning algorithm you find appropriate. 
# learn structure of our BN
trained_dag = hc(x = traindata)
graphviz.plot(trained_dag, main = 'trained_dag')

# fit the parameters of our BN 
# i.e. gives us all the conditiotnal probability tables for all our parameters and their 
# probabilities of being yes/no
fitted = bn.fit(x = trained_dag, traindata)
#fitted

# converty bn network to gRain object - Graphical Independence Network
# grain is an implementation of exact inference in R for discrete BNs. 
# the gRain class object stores a fitteed BN as a list of conditional probability tables. 
grain = as.grain(fitted)
# Compiled: TRUE, objects are compiled upon creation by default.- 
# Compiling means creating a junction tree and establishing clique potentials
grain
# gives us the cliques, separates, and nr of parents for each node. 
grain$rip

# plots the gRain graph. 
plot(grain)

# Use the BN to classify the remaining 20% of the asia dataset into 2 classes: S = yes, adn S = no. 
# I.e.: Compute posterior probability distribution of S for each case, and classify it in the most likely class. 
# To do so, you have to use EXACT or approximate inference with the help of the bnlearn and gRain packages. 

# setEvidence - set, update and remove evidence of grain object. 
# i.e. we can update our grain oject with new evidence. 
# Query an independence network, i.e. obtain the conditional distribution of a set of variables - 
# possibly (and typically) given finding (evidence) on other variables.


predict_S = function(bn, data, observed_nodes) {
  # vector for storing predictions
  predictions = c(1:nrow(data))
  for (i in 1:nrow(data)) {
    # for each of our observations, we have to update the known values
    # i.e. the evidence, on A, T, L, B, E, X, D - i.e. all nodes except S, since we want to predict S with our BN.
    # we can do this with setEvidence to update our grain object.
    
    # extract row i and remove column S
    #case =  subset(data[i,], select = -c(S))
    case =  subset(data[i,], select = observed_nodes)
    
    # store observed states for the case
   # observed_states = c(1:ncol(case))
    observed_states = NULL
    for (state in 1:ncol(case)) {
      if (case[state] == 'yes') {
        observed_states = c(observed_states, 'yes')
      }
      else {
        observed_states = c(observed_states, 'no')
      }
    }
    
    # predict S using our observed states
    probabilities = querygrain(object = setEvidence(bn, nodes = observed_nodes, 
                            states = observed_states), 
                            nodes = c("S"), 
                            type="joint")
    
    if (probabilities['yes'] > 0.5) {
      predictions[i] = 'yes'
    }
    else {
      predictions[i] = 'no'
    }
  }
  return(predictions)
}

predictions = predict_S(grain, testdata, c("A", "T", "L", "B", "E", "X", "D"))

# Report the confusion matrix. 
cm = table(predictions, testdata$S)
cm

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



#################### ASSIGNMENT 3 #################### 
# get the markov blanket from the fitted BN, 
# These are the observed nodes we will use for classifying S
markov_blanket = mb(fitted, 'S')
markov_blanket

predictions_mb = predict_S(grain, testdata, markov_blanket)

cm_mb = table(predictions_mb, testdata$S)
cm_mb

# Compare result with the true Asia BN:
markov_blanked_true = mb(fitted_true, 'S')
predictions_mb_true = predict_S(grain_true, testdata, markov_blanked_true)
cm_mb_true = table(predictions_mb_true, testdata$S)
cm_mb_true

#################### ASSIGNMENT 4 #################### 

# like in 2, learn structure and parameters,
# using naive bayes classifier. 
# predictive variables are independent given the class variable S!

# lets look in the graph. the predictive variables should be indepoendent given the class variable S!!!!!!!
graphviz.plot(model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]"), main = 'true DAG')
graphviz.plot(trained_dag, main = 'trained DAG') 

naive_dag <- model2network("[S][A|S][L|S][T|S][E|S][B|S][X|S][D|S]")

graphviz.plot(naive_dag, main = 'true DAG')
fitted_naive = bn.fit( x = naive_dag, traindata)
grain_naive = as.grain(fitted_naive)
grain_naive
grain_naive$rip
plot(grain_naive)

predictions_naive = predict_S(grain_naive, testdata, c("A", "T", "L", "B", "E", "X", "D"))

cm_naive = table(predictions_naive, testdata$S)
cm_naive




#################### ASSIGNMENT 5 #################### 
# Explain why you obtain the same or different results in the exercises (2-4).
# 2-3 same. so markov belt is enough to predict. dont need rest, since all information that the markov belt
# carries is enough if we know it - i.e. the additional nodes, which affect the markov belt, are not needed, 
# since the information from them is implicily encoded into the nodes in the markov belt.
# thus it is enough if we ovserve the nodes in the markov belt. 



# 4 different, worse. due to naive approach?



