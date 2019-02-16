#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import math

    cleaned_data = []

    sortedError = []
    for i in range(len(predictions)):
        sortedError.extend(abs(predictions[i] - net_worths[i]))

    sortedError.sort()

    keeperCount = int(math.floor(len(sortedError)*0.9))

    topErrors = sortedError[keeperCount:]

    for i in range(len(predictions)):
        error = abs(predictions[i] - net_worths[i])
        if error in topErrors:
            pass
        else:
            cleaned_data.append([ages[i], net_worths[i], error])

    return cleaned_data

