from basicFunction import * 


def bgd(activation, activation_d, lossFunction, lossFunction_d, weight_init_func, x, y, epochs = 99999, learningRate = 0.001, hiddenLayerSize = 25):

    weight1, weight2 = weight_init_func(len(x[0]), hiddenLayerSize)

    for epoch in range(epochs):
        n_deltas2 = np.zeros_like(weight2)
        n_deltas1 = np.zeros_like(weight1)
        
        hiddenVals = np.dot(x, weight1) # m * k
        activeVals = activation(hiddenVals) 
        activeVals = np.hstack([activeVals, np.ones((activeVals.shape[0], 1))]) # m * k + 1
        outputVals = np.dot(activeVals, weight2) # m * 1
        y_hat = (outputVals) # m * 1 

        loss = lossFunction(y, y_hat)
        if epoch % 100 == 0:
            print(f"epoch {epoch:6d} | loss = {loss:.11f}")
        
        n_deltas2 = activeVals.T @ lossFunction_d(y, y_hat) 
        n_deltas1 = x.T @ (lossFunction_d(y, y_hat) * (weight2.T[:, :-1] * activation_d(hiddenVals)))
        weight2 -= learningRate * n_deltas2 / len(x)
        weight1 -= learningRate * n_deltas1 / len(x)

    return weight1, weight2

def mbgd(activation, activation_d, lossFunction, lossFunction_d, weight_init_func, x, y, epochs = 99999, learningRate = 0.001, hiddenLayerSize = 25, batch_size = 64):

    weight1, weight2 = weight_init_func(len(x[0]), hiddenLayerSize)
    
    samples = np.append(x, y, axis=1)

    for epoch in range(epochs):
     
        np.random.shuffle(samples)

        loss = 0
        for start in range(0, len(samples), batch_size):
            end = start + batch_size
            x_batch = samples[start:end, :-1]
            y_batch = samples[start:end, -1:]

            n_deltas2 = np.zeros_like(weight2)
            n_deltas1 = np.zeros_like(weight1)
            
            hiddenVals = np.dot(x_batch, weight1) # m * k
            activeVals = activation(hiddenVals) 
            activeVals = np.hstack([activeVals, np.ones((activeVals.shape[0], 1))]) # m * k + 1
            outputVals = np.dot(activeVals, weight2) # m * 1
            y_hat = (outputVals) # m * 1 

            loss += lossFunction(y_batch, y_hat)

            n_deltas2 = activeVals.T @ lossFunction_d(y_batch, y_hat) 
            n_deltas1 = x_batch.T @ (lossFunction_d(y_batch, y_hat) * (weight2.T[:, :-1] * activation_d(hiddenVals)))
            weight2 -= learningRate * n_deltas2 / len(x_batch)
            weight1 -= learningRate * n_deltas1 / len(x_batch)


        loss /= (len(samples) / batch_size)
        if epoch % 100 == 0:
            print(f"epoch {epoch:6d} | loss = {loss:.11f}")


    return weight1, weight2

def adam_mbgd(activation, activation_d, lossFunction, lossFunction_d, weight_init_func, x, y, epochs = 99999, learningRate = 0.001, hiddenLayerSize = 25, batch_size = 64, beta1 = 0.9, beta2 = 0.999):

    t = 0
    m1, m2 = 0, 0 
    v1, v2 = 0, 0
    epsilon = np.finfo(np.float64).eps

    weight1, weight2 = weight_init_func(len(x[0]), hiddenLayerSize)
    
    samples = np.append(x, y, axis=1)

    for epoch in range(epochs):
        
        np.random.shuffle(samples)

        loss = 0
        for start in range(0, len(samples), batch_size):
            t += 1
            
            end = start + batch_size
            x_batch = samples[start:end, :-1]
            y_batch = samples[start:end, -1:]

            n_deltas2 = np.zeros_like(weight2)
            n_deltas1 = np.zeros_like(weight1)
            
            hiddenVals = np.dot(x_batch, weight1) # m * k
            activeVals = activation(hiddenVals) 
            activeVals = np.hstack([activeVals, np.ones((activeVals.shape[0], 1))]) # m * k + 1
            outputVals = np.dot(activeVals, weight2) # m * 1
            y_hat = (outputVals) # m * 1 

            loss += lossFunction(y_batch, y_hat)

            n_deltas2 = activeVals.T @ lossFunction_d(y_batch, y_hat) 
            n_deltas1 = x_batch.T @ (lossFunction_d(y_batch, y_hat) * (weight2.T[:, :-1] * activation_d(hiddenVals)))

            n_deltas1 /= len(x_batch)
            n_deltas2 /= len(x_batch)

            m1 = beta1 * m1 + (1 - beta1) * n_deltas1
            v1 = beta2 * v1 + (1 - beta2) * n_deltas1 ** 2
            m1_hat = m1 / (1 - beta1 ** t)
            v1_hat = v1 / (1 - beta2 ** t)
            weight1 -= (learningRate / (np.sqrt(v1_hat) + epsilon)) * m1_hat

            m2 = beta1 * m2 + (1 - beta1) * n_deltas2
            v2 = beta2 * v2 + (1 - beta2) * n_deltas2 ** 2
            m2_hat = m2 / (1 - beta1 ** t)
            v2_hat = v2 / (1 - beta2 ** t)
            weight2 -= (learningRate / (np.sqrt(v2_hat) + epsilon)) * m2_hat



        loss /= (len(samples) / batch_size)
        if epoch % 100 == 0:
            print(f"epoch {epoch:6d} | loss = {loss:.11f}")


    return weight1, weight2
