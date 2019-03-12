# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:58:32 2019

@author: Vivek
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from loadFoodData import LoadFoods
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):
    
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        
    def computeCuisineSimilarity(self, food1, food2, cuisine):
        cuisine1 = cuisine[food1]
        cuisine2 = cuisine[food2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(cuisine1)):
            x = cuisine1[i]
            y = cuisine2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def computeOrderTimeSimilarity(self, food1, food2, years):
        diff = abs(years[food1] - years[food2])
        sim = math.exp(-diff / 10.0)
        return sim
    
    def instaFit(self,trainset):
        AlgoBase.fit(self,trainset)
        return self
    
    
    def fit(self,trainset):
        AlgoBase.fit(self,trainset)
        
        lf = LoadFoods()
        cuisines = lf.getCuisines()
        #orderTime = lf.getOrderTime()
        
        print("Now computing content-based similarity matrix. Please wait ...")
        
        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRating in range(self.trainset.n_items):
            if ( thisRating%100 == 0 and thisRating != 0 ):
                print("processed ", thisRating, " of ", self.trainset.n_items, " items")
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisFoodID = ( self.trainset.to_raw_iid(thisRating))
                otherFoodID = ( self.trainset.to_raw_iid(thisRating))
                cuisineSimilarity = self.computeCuisineSimilarity(thisFoodID,otherFoodID,cuisines)
                #orderTimeSimilarity = self.computeOrderTimeSimilarity(thisFoodID,otherFoodID,orderTime)
                self.similarities[thisRating,otherRating] = cuisineSimilarity
                self.similarities[otherRating,thisRating] = self.similarities[thisRating,otherRating]
                
        
        print("done computing the matrix...")
        return self
    
                
    def estimate(self, u, i):
        #if user or item is not present in trainset then throw the exception
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            cuisineSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (cuisineSimilarity, rating[1]) )
        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
        # throw an exception if there is no neighbors for this user
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        