# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:39:11 2019

@author: Vivek
"""

from loadFoodData import LoadFoods
from contentKNNalgoBase import ContentKNNAlgorithm
import heapq


lf = LoadFoods()
dataset = lf.loadFoodData()
trainset = dataset.build_full_trainset()

contentKNN = ContentKNNAlgorithm()
contentKNN.fit(trainset)

test_user = 85
k = 10


user_ratings = lf.getUserRatings(test_user)
predictions = {}

for foodID,ratings in user_ratings:
    predicted_rating = contentKNN.estimate(test_user,foodID)
    foodName = lf.getFoodName(foodID)
    predictions[foodName] = predicted_rating

top_k_predictions = heapq.nlargest(k,predictions, key = lambda x: predictions[x])
print('now predicting ....')
for prediction in top_k_predictions:
    print(prediction)
