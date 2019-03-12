# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:34:26 2019

@author: Vivek
"""

from loadFoodData import LoadFoods
from tester import ContentKNNAlgorithm
from surprise import SVD
import heapq
import numpy as np
import random

def getAllFoodID():
    cuisineList = lf.getCuisines()
    foodIDList = []
    for foodID in cuisineList:
        foodIDList.append(foodID)
    return foodIDList

def getAntiUserFoodID(user):
    allFoodID = getAllFoodID()
    user_Ratings = lf.getUserRatings(int(user))
    userRatedFoodID = []
    userNotRatedFoodID = []
    for ur in user_Ratings:
        userRatedFoodID.append(ur[0])
    for foodID in allFoodID:
        if not int(foodID) in userRatedFoodID:
            userNotRatedFoodID.append(int(foodID))
    
    return userNotRatedFoodID


lf = LoadFoods()
data = lf.loadFoodData()
trainset = data.build_full_trainset()

np.random.seed(0)
random.seed(0)

print('training the model ....')
SVD = SVD()
SVD.fit(trainset)

test_user = '85'
k = 10
test_user_innerID = trainset.to_inner_uid(test_user)

userNRFID = getAntiUserFoodID(test_user_innerID)

pred_ratings = {}
for foodID in userNRFID:
    pred = SVD.estimate(test_user_innerID,foodID)
    foodName = lf.getFoodName(foodID)
    pred_ratings[foodName] = pred

top_k_predictions = heapq.nlargest(k,pred_ratings,key = lambda x: pred_ratings[x])

print('now predicting ....')
for prediction in top_k_predictions:
    print(prediction)
    

