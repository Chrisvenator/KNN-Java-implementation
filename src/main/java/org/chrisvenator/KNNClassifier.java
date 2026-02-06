package org.chrisvenator;

import org.chrisvenator.distance.DistanceMetric;
import org.chrisvenator.distance.EuclideanDistance;

public class KNNClassifier {
    private int k;
    private DistanceMetric distanceMetric;
    private double[][] trainingData;
    private int[] labels;
    
    public KNNClassifier(int k) {
        this(k, new EuclideanDistance());
    }
    
    public KNNClassifier(int k, DistanceMetric distanceMetric) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive!");
        }
        
        this.k = k;
        this.distanceMetric = distanceMetric;
    }
    
    /**
     * Train the classifier with data
     *
     * @param trainingData 2D array where each row is a sample, each column a feature
     * @param labels       Array of integer labels for each training sample
     */
    public void fit(double[][] trainingData, int[] labels) {
        this.trainingData = trainingData;
        this.labels = labels;
    }
    
    /**
     * Predict the class of a single test point
     *
     * @param testPoint Feature vector to classify
     * @return Predicted class label
     */
    public int predict(double[] testPoint) {
        return -1;
    }
    
    public int predict(int k, double[] testPoint) {
        int tempK = this.k;
        this.k = k;
        
        int predict = predict(testPoint);
        this.k = tempK;
        
        return predict;
    }
}


























