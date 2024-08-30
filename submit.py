import numpy as np
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y_train contains the responses
	
    feat_train = my_map(X_train)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(feat_train, y_train)
    
    # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
    # If you do not wish to use a bias term, set it to 0
    w = classifier.coef_.T.flatten()  # Flatten the weights
    b = classifier.intercept_
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( C ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to create features.
    # It is likely that my_fit will internally call my_map to create features for train points
    D = 1 - 2 * C
    X = np.flip(np.cumprod(np.flip(D, axis=1), axis=1), axis=1)
    feat = []
    for x in X:
        x_reshaped = x.reshape(-1, 1)
        y = x_reshaped * x
        y[np.tril_indices_from(y)] = 0
        y = y.flatten()
        y = np.concatenate((y[y != 0], x))
        feat.append(y)
    feat = np.array(feat)
    return feat
