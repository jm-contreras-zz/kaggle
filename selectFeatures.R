selectFeatures = function(x, y, guide, reg, param) {

    # Load package
    library(RRF)
        
    # Regularized forest
    if (guide == 0) {
        
        lambda = param  # Small lambda = Fewer features
        rrf = RRF(x, y, flagReg = reg, coefReg = lambda)
        imp = rrf$importance[, "MeanDecreaseGini"]
    
    # Guided (regularized) random forest
    } else if (guide == 1) {
        
        rf = RRF(x, y, flagReg = reg)
        imp.rf = rf$importance[, "MeanDecreaseGini"]
        imp = imp.rf / (max(imp.rf))  # Normalize importance score
        gamma = param  # Large gamma = Fewer features
        coef.reg = (1 - gamma) + gamma * imp  # Weighted average
        grf = RRF(x, y, coefReg = coef.reg, flagReg = reg)
        imp = grf$importance[, "MeanDecreaseGini"]
        
    }
    
    # Return selected features with importance scores
    return(names(imp[imp > 0]))
    
}
