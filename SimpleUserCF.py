# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:34:26 2019

@author: Vivek
"""

from loadFoodData import LoadFoods
from surprise import KNNBasic

import heapq
from collections import defaultdict
from operator import itemgetter

testSubject = '85'
k = 10

lf = LoadFoods()
dataset = lf.loadFoodData()

trainSet = dataset.build_full_trainset()

sim_options = {'name': 'cosine',
              'user_based': True
              }

model = KNNBasic(sim_options=sim_options)
sm = model.fit(trainSet)
simsMatrix = model.compute_similarities()


# Get top N similar users to our test subject
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow): #enumerate it to get the index ,score of each cell in similarityRow
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
    
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        foodID = trainSet.to_raw_iid(itemID)
        print(lf.getFoodName(int(foodID)), ratingSum)
        pos += 1
        if (pos > 10):
            break
    