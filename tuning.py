def markov_chain_monte_carlo(iterations, model, X, Y, step_size=0.1, folds=10, burn_period = 500, score='accuracy'):

    '''

    :param iterations: # of chains in the MCMC to run (typically between 10,000-20,000 depending on model)
    :param model: the model object to run on
    :param X: the feature_space
    :param Y: the target data
    :param step_size: the standard deviation of a normal distribution that determines how "large" the proposed next parameters are
    :param burn_period: # of chains to discards from beginning of MCMC
    :return: array showing parameter step history
    '''

    step_history = []
    parameter = 1 #initialize at 1 for linearSVC regularization
    assert(burn_period < iterations) #must be less than the total iterations

    for chain in range(iterations):

        #model = OneVsRestClassifier(model(random_state=0)).fit(X, Y)
        model.C = parameter
        score = np.mean(cross_val_score(model, X, Y, cv=folds, scoring='accuracy'))
        score = np.mean(cross_val_score(mlPerceptron, X, Y, cv=folds, scoring='accuracy'))
        print("Iteration {0}, kappa score: {1}".format(chain, score))
        print("Parameter location: {0}".format(model.C))
        threshold = numpy.random.uniform(0,1)
        print("Threshold: {0}".format(threshold))

        if score > threshold:
            #reject proposal
            step_history.append(parameter)
            print("\n***Proposal rejected***")

        else:
            #accept proposal
            print("\n***Proposal accepted***")

            proposal_step = numpy.random.normal(0, step_size)
            parameter = proposal_step + parameter
            #update the model's parameter and iterate in next chain of MCMC
            step_history.append(parameter)

    return step_history[-(iterations-burn_period):] #account for burn in rate

markov_chain_monte_carlo(2000, linearSVC, X=cluster_center_list, Y=label_cluster_list, burn_period=200,  folds=4, step_size=.5)