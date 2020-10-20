# Conclusions

In terms of cleaning the data, we used the pandas get_dummies feature to separate the nominal feature of race to a 
one-hot encoded vector, which we added as a feature. We then moved our y vector, "Chance of Admit", to the end. For more
notes on what we did, see the Cleaner Notebook. 

In conclusion, after taking a deep dive into each of the models, we found the model with the best accuracy
was the SKLearn implementation of SVM, with the hyperparameters kernel="linear" and C=1, which
gave us an accuracy of 93%. Almost all of our models, save linear regression, were consistently in the mid 80s
in terms of accuracy. However, the SVM was the only model to consistently have accuracy scores over 90%.